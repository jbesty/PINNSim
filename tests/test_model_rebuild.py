from pinnsim.post_processing.rebuild_nn_model import rebuild_trained_nn_model


def test_model_rebuild():

    run_id = "gc0gm229"
    rebuild_trained_nn_model(run_id=run_id)
    pass
