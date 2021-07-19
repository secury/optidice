import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network, utils


class ValueNetwork(network.Network):

    def __init__(self, input_tensor_spec, hidden_sizes, output_activation_fn=None, last_layer_bias=None, output_dim=None, name='ValueNetwork'):
        """Create an instance of `ValueNetwork`.
        - hidden_sizes = (256, 256)
        """
        super(ValueNetwork, self).__init__(input_tensor_spec, state_spec=(), name=name)

        self._input_specs = input_tensor_spec
        self._output_dim = output_dim

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu, kernel_initializer='glorot_uniform', name='mlp')
        last_layer_initializer = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
        self._last_layer = tf.keras.layers.Dense(output_dim or 1, activation=output_activation_fn, kernel_initializer=last_layer_initializer, bias_initializer=last_layer_bias or last_layer_initializer, name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        del step_type  # unused
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)
        h = self._last_layer(h)

        if self._output_dim is None:
            h = tf.reshape(h, [-1])

        return h, network_state


class TanhNormalPolicy(network.Network):

    def __init__(self, input_tensor_spec, action_dim, hidden_sizes, name='TanhNormalPolicy',
                 mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=1e-6):
        super(TanhNormalPolicy, self).__init__(input_tensor_spec, state_spec=(), name=name)

        self._input_specs = input_tensor_spec  # (obs_spec)
        self._action_dim = action_dim

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu, kernel_initializer='glorot_uniform', name='mlp')
        last_layer_initializer = tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
        self._fc_mean = tf.keras.layers.Dense(action_dim, name='policy_mean/dense', kernel_initializer=last_layer_initializer, bias_initializer=last_layer_initializer)
        self._fc_logstd = tf.keras.layers.Dense(action_dim, name='policy_logstd/dense', kernel_initializer=last_layer_initializer, bias_initializer=last_layer_initializer)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, step_type=(), network_state=(), training=False):
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd)
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_action = pretanh_action_dist.sample()
        action = tf.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(tf.clip_by_value(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - action ** 2 + self.eps), axis=-1)

        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=False)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        action = tf.tanh(mean)

        return action


class TanhMixtureNormalPolicy(network.Network):

    def __init__(self, input_tensor_spec, action_dim, hidden_sizes, num_components=2, name='TanhMixtureNormalPolicy',
                 mean_range=(-9., 9.), logstd_range=(-5., 2.), eps=1e-6, mdn_temperature=1.0):
        super(TanhMixtureNormalPolicy, self).__init__(input_tensor_spec, state_spec=(), name=name)

        self._input_specs = input_tensor_spec  # (obs_spec)
        self._action_dim = action_dim
        self._num_components = num_components
        self._mdn_temperature = mdn_temperature

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu, kernel_initializer='glorot_uniform', name='mlp')
        last_layer_initializer = tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
        self._fc_means = tf.keras.layers.Dense(num_components * action_dim, name='policy_mean/dense', kernel_initializer='glorot_uniform', bias_initializer=last_layer_initializer)
        self._fc_logstds = tf.keras.layers.Dense(num_components * action_dim, name='policy_logstd/dense', kernel_initializer=last_layer_initializer, bias_initializer=last_layer_initializer)
        self._fc_logits = tf.keras.layers.Dense(num_components, name='policy_logits/dense', kernel_initializer='glorot_uniform', bias_initializer=last_layer_initializer)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, step_type=(), network_state=(), training=False):
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)

        means = self._fc_means(h)
        means = tf.clip_by_value(means, self.mean_min, self.mean_max)
        means = tf.reshape(means, (-1, self._num_components, self._action_dim))
        logstds = self._fc_logstds(h)
        logstds = tf.clip_by_value(logstds, self.logstd_min, self.logstd_max)
        logstds = tf.reshape(logstds, (-1, self._num_components, self._action_dim))
        stds = tf.exp(logstds)

        component_logits = self._fc_logits(h) / self._mdn_temperature

        pretanh_actions_dist = tfp.distributions.MultivariateNormalDiag(means, stds)
        component_dist = tfp.distributions.Categorical(logits=component_logits)

        pretanh_actions = pretanh_actions_dist.sample()  # (batch_size, num_components, action_dim)
        component = component_dist.sample()  # (batch_size)

        batch_idx = tf.range(tf.shape(inputs[0])[0])
        pretanh_action = tf.gather_nd(pretanh_actions, tf.stack([batch_idx, component], axis=1))
        action = tf.tanh(pretanh_action)

        log_prob, pretanh_log_prob = self.log_prob((component_dist, pretanh_actions_dist), pretanh_action, is_pretanh_action=True)

        return (action, pretanh_action, log_prob, pretanh_log_prob, (component_dist, pretanh_actions_dist)), network_state

    def log_prob(self, dists, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(tf.clip_by_value(action, -1 + self.eps, 1 - self.eps))

        component_dist, pretanh_actions_dist = dists
        component_logits = component_dist.logits_parameter()
        component_log_prob = component_logits - tf.math.reduce_logsumexp(component_logits, axis=-1, keepdims=True)

        pretanh_actions = tf.tile(pretanh_action[:, None, :], (1, self._num_components, 1))  # (batch_size, num_components, action_dim)

        pretanh_log_prob = tf.reduce_logsumexp(component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions), axis=1)
        log_prob = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - action ** 2 + self.eps), axis=-1)

        return log_prob, pretanh_log_prob
