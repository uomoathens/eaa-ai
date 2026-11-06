def responsible_reward_original(utility, bias_loss, c_ethics, gamma_bias, gamma_ethics):
    return float(utility - gamma_bias * bias_loss + gamma_ethics * c_ethics)


def responsible_reward_refined(utility, bias_loss, hinge_penalty, gamma_bias, gamma_ethics):
    return float(utility - gamma_bias * bias_loss - gamma_ethics * hinge_penalty)
