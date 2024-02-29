ieee9_H = [23.64, 6.4, 3.01]
ieee9_D = [H * D_H_ratio for H, D_H_ratio in zip(ieee9_H, [0.1, 0.2, 0.3])]
ieee9_X_d_prime_pu = [0.0608, 0.1198, 0.1813]
ieee9_omega_scale = [0.0500, 0.0690, 0.0811]


def get_machine_data(seed=None):
    assert seed in ["ieee9_1", "ieee9_2", "ieee9_3"]
    case, gen_id_str = seed.split("_")
    gen_id = int(gen_id_str)

    name = f"IEEE 9-bus Gen{gen_id}"
    H_s = ieee9_H[gen_id - 1]
    D_pu = ieee9_D[gen_id - 1]
    X_d_prime_pu = ieee9_X_d_prime_pu[gen_id - 1]
    norm_to_scale_delta = 1.7059
    norm_to_scale_omega = ieee9_omega_scale[gen_id - 1]

    machine_data = dict(
        {
            "generator_name": seed,
            "parameter_set_name": name,
            "H_s": H_s,
            "D_pu": D_pu,
            "X_d_pu": X_d_prime_pu,
            "X_d_prime_pu": X_d_prime_pu,
            "X_q_pu": X_d_prime_pu,
            "X_q_prime_pu": X_d_prime_pu,
            "T_d0_prime_s": 8.96,
            "T_q0_prime_s": 0.31,
            "R_s_pu": 0.0,
            "norm_to_scale_delta": norm_to_scale_delta,
            "norm_to_scale_omega": norm_to_scale_omega,
        }
    )

    return machine_data
