from pinnsim.configurations.dataset_config import define_dataset_config


def test_answer():
    dataset_config = define_dataset_config(
        dataset_type="test", generator_id="ieee9_1", seed=91249045
    )
    assert dataset_config["n_operating_points"] == 4000
