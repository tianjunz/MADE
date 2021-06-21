import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra
import kornia
import os

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""

    def __init__(self, image_size, feature_dim, k, channel):
        super().__init__()

        self.num_layers = 4
        self.num_filters = 32
        self.channel = channel
        if image_size == 84:
            self.output_dim = 35
        elif image_size in range(84, 122, 2):
            self.output_dim = 35 + (image_size - 84) // 2
        else:
            raise ValueError(image_size)

        self.output_logits = False
        self.feature_dim = feature_dim
        self.k = k

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.channel, self.num_filters, 3, stride=2),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(
                self.num_filters * self.output_dim * self.output_dim, self.feature_dim
            ),
            nn.LayerNorm(self.feature_dim),
        )

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs["out"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_encoder/{k}_hist", v, step)
            if len(v.shape) > 2:
                logger.log_image(f"train_encoder/{k}_img", v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f"train_encoder/conv{i + 1}", self.convs[i], step)

    def compute_state_entropy(self, src_feats, tgt_feats, average_entropy=False):
        with torch.no_grad():
            dists = []
            for idx in range(len(tgt_feats) // 10000 + 1):
                start = idx * 10000
                end = (idx + 1) * 10000
                dist = torch.norm(
                    src_feats[:, None, :] - tgt_feats[None, start:end, :], dim=-1, p=2
                )
                dists.append(dist)

            dists = torch.cat(dists, dim=1)
            knn_dists = 0.0
            if average_entropy:
                for k in range(5):
                    knn_dists += torch.kthvalue(dists, k + 1, dim=1).values
                knn_dists /= 5
            else:
                knn_dists = torch.kthvalue(dists, k=self.k + 1, dim=1).values
            state_entropy = knn_dists
        return state_entropy.unsqueeze(1)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self, encoder_cfg, action_shape, hidden_dim, hidden_depth, log_std_bounds
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(
            self.encoder.feature_dim, hidden_dim, 2 * action_shape[0], hidden_depth
        )

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_actor/{k}_hist", v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f"train_actor/fc{i}", m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(
            self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth
        )
        self.Q2 = utils.mlp(
            self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth
        )

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f"train_critic/{k}_hist", v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f"train_critic/q1_fc{i}", m1, step)
                logger.log_param(f"train_critic/q2_fc{i}", m2, step)

# RND Predictor
class Predictor(nn.Module):
    def __init__(self, encoder_cfg, hidden_dim, hidden_depth):
        super().__init__()
        self.mlp = utils.mlp(encoder_cfg.params.feature_dim, hidden_dim, 128, hidden_depth)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        feature = self.mlp(obs)
        self.outputs["feature"] = feature
        return feature

# RND Target
class Target(nn.Module):
    def __init__(self, encoder_cfg, hidden_dim, hidden_depth):
        super().__init__()
        self.mlp = utils.mlp(encoder_cfg.params.feature_dim, hidden_dim, 128, hidden_depth)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        feature = self.mlp(obs)
        self.outputs["feature"] = feature
        return feature

# VAE Network
class VAE(nn.Module):
    def __init__(self, image_size, feature_dim, channel):
        super().__init__()

        self.num_layers = 4
        self.num_filters = 32
        self.channel = channel
        if image_size == 84:
            self.output_dim = 35
        elif image_size in range(84, 122, 2):
            self.output_dim = 13
        else:
            raise ValueError(image_size)

        self.output_logits = False
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, self.num_filters, 3, stride=2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters*2, 3, stride=2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters*2, self.num_filters*4, 3, stride=2, padding  = 1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(
                self.num_filters * 4 * self.output_dim * self.output_dim, self.feature_dim
            ),
        )
        self.fc_var = nn.Sequential(
            nn.Linear(
                self.num_filters * 4 * self.output_dim * self.output_dim, self.feature_dim 
            ),
        )
         
        self.decode_fc = nn.Sequential(
            nn.Linear(
                self.feature_dim, self.num_filters * 4 * self.output_dim * self.output_dim
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters*4, self.num_filters*2,
                                    kernel_size=3,
                                    stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters*2, self.num_filters,
                                    kernel_size=3,
                                    stride = 2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters, self.channel,
                                    kernel_size=3,
                                    stride = 2, padding=1, output_padding=0),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(nn.Conv2d(self.channel, self.channel, 2, stride=1, padding=0), nn.Tanh())

        self.outputs = dict()

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decode_fc(z)
        result = result.view(-1, self.num_filters*4, self.output_dim, self.output_dim)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        input[:, :3] = input[:, :3] / 255.0
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        ir = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) + torch.mean(F.mse_loss(recons, input, reduction='none'), dim=(-3, -2, -1))
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss, 'ir':ir}


class MADEAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        action_range,
        device,
        encoder_cfg,
        critic_cfg,
        actor_cfg,
        random_encoder_cfg,
        discount,
        init_temperature,
        lr,
        alpha_lr,
        actor_update_frequency,
        critic_tau,
        encoder_tau,
        critic_target_update_frequency,
        batch_size,
        image_size,
        use_rnd,
        normalize_rnd,
        average_rnd,
        beta_schedule,
        beta_init,
        beta_decay,
        aug_type,
        use_drq,
        vae_iter,
    ):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.image_size = image_size
        self.aug_type = aug_type
        self.use_drq = use_drq
        self.vae_iter = vae_iter

        self.crop = kornia.augmentation.RandomCrop((image_size, image_size))
        self.center_crop = kornia.augmentation.CenterCrop((image_size, image_size))

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # random encoder
        self.random_encoder = hydra.utils.instantiate(random_encoder_cfg).to(self.device) 
        self.predictor = Predictor(random_encoder_cfg, 512, 3).to(
            self.device
        )
        self.target = Target(random_encoder_cfg, 512, 1).to(
            self.device
        )
        # vae
        self.vae = VAE(100, 128, 3 + action_shape[0]).to(self.device)
        self.input_stats = utils.TorchRunningMeanStd(shape=[1, self.random_encoder.channel, 100, 100], device=device)

        # state entropy
        self.use_rnd = use_rnd
        self.normalize_rnd = normalize_rnd
        self.average_rnd = average_rnd
        self.beta_schedule = beta_schedule
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.rnd_stats = utils.TorchRunningMeanStd(shape=[1], device=device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=alpha_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.predictor.train(training)
        self.vae.train(training)
        self.target.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def beta(self, step):
        if self.beta_schedule == "constant":
            return self.beta_init
        elif self.beta_schedule == "linear_decay":
            return self.beta_init * ((1 - self.beta_decay) ** step)
        else:
            raise ValueError(self.beta_schedule)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        if self.aug_type == "crop":
            obs = self.center_crop(obs)
        elif self.aug_type == "translate":
            pad = (self.image_size - obs.shape[-1]) // 2
            obs = F.pad(obs, [pad, pad, pad, pad])
        else:
            raise ValueError(self.aug_type)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(
        self,
        obs,
        obs_aug,
        raw_obs,
        action,
        reward,
        next_obs,
        next_obs_aug,
        raw_next_obs,
        not_done,
        src_feat,
        tgt_feat,
        logger,
        step,
    ):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

            channel = self.random_encoder.channel
            embed = self.random_encoder(raw_next_obs[:, -channel:]).detach()
            target = self.target(embed)
            predictor = self.predictor(embed)
            rnd = torch.norm(target.detach()-predictor.detach(), dim=-1, p=2).unsqueeze(-1)

            inputs = torch.cat([raw_obs[:, -channel:], action.view(action.shape[0], -1, 1, 1).repeat(1, 1, raw_obs.shape[2], raw_obs.shape[3])], dim=1)
            results = self.vae.forward(inputs)
            vae_loss = self.vae.loss_function(*results, M_N = 1)['ir'].detach()
            vae_intrinsic = torch.exp(-vae_loss)**(-0.5)
            rnd *= vae_intrinsic.unsqueeze(-1)

            logger.log("train_critic/rnd", rnd.mean(), step)
            logger.log("train_critic/rnd_max", rnd.max(), step)
            logger.log("train_critic/rnd_min", rnd.min(), step)

            self.rnd_stats.update(vae_intrinsic)
            norm_rnd = rnd / self.rnd_stats.std

            logger.log("train_critic/norm_rnd", norm_rnd.mean(), step)
            logger.log("train_critic/norm_rnd_max", norm_rnd.max(), step)
            logger.log("train_critic/norm_rnd_min", norm_rnd.min(), step)

            if self.normalize_rnd:
                int_reward = norm_rnd
            else:
                int_reward = rnd

            target_Q = (
                reward
                + self.beta(step) * int_reward
                + (not_done * self.discount * target_V)
            )

            if self.use_drq:
                dist_aug = self.actor(next_obs_aug)
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
                target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
                target_V = (
                    torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                )
                target_Q_aug = (
                    reward
                    + self.beta(step) * int_reward
                    + (not_done * self.discount * target_V)
                )

                target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        if self.use_drq:
            Q1_aug, Q2_aug = self.critic(obs_aug, action)
            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the rnd
        with torch.no_grad():
            embed = self.random_encoder(raw_next_obs[:, -channel:]).detach()
        target = self.target(embed)
        predictor = self.predictor(embed)
        rnd_loss = F.mse_loss(predictor, target.detach())

        self.predictor_optimizer.zero_grad()
        rnd_loss.backward()
        self.predictor_optimizer.step()

        self.critic.log(logger, step)

    def update_vae(self, l_raw_obs, l_action, l_raw_next_obs, rnd_raw_obs, update_rnd):
        # Optimize the vae
        channel = self.random_encoder.channel
        self.input_stats.update(l_raw_obs[:, -channel:])

        inputs = torch.cat([l_raw_obs[:, -channel:], l_action.view(l_action.shape[0], -1, 1, 1).repeat(1, 1, l_raw_obs.shape[2],l_raw_obs.shape[3])], dim=1)
        results = self.vae.forward(inputs)
        vae_loss = self.vae.loss_function(*results, M_N = 1)['loss']
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        logger.log("train_alpha/loss", alpha_loss, step)
        logger.log("train_alpha/value", self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, latest_replay_buffer, logger, step):
        (
            obs,
            obs_aug,
            raw_obs,
            action,
            reward,
            next_obs,
            next_obs_aug,
            raw_next_obs,
            not_done,
            src_feat,
            tgt_feat,
        ) = replay_buffer.sample(self.batch_size)
       

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(
            obs,
            obs_aug,
            raw_obs,
            action,
            reward,
            next_obs,
            next_obs_aug,
            raw_next_obs,
            not_done,
            src_feat,
            tgt_feat,
            logger,
            step,
        )
        for _ in range(self.vae_iter):
            rnd_raw_obs = None
            update_rnd = False
            (_, _, l_raw_obs, l_action, _, _, _, l_raw_next_obs, _, _, _) = latest_replay_buffer.sample(128)
            self.update_vae(
                l_raw_obs,
                l_action,
                l_raw_next_obs,
                rnd_raw_obs,
                update_rnd
            )

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

    def save(self, save_dir, step):
        torch.save(
            self.actor.state_dict(), os.path.join(save_dir, "actor_%d.pt" % (step))
        )
        torch.save(
            self.critic.state_dict(), os.path.join(save_dir, "critic_%d.pt" % (step))
        )

