import pathlib
import pickle


def save_dataset_raw(dataset_raw, dataset_name, data_path=None):
    if data_path is None:
        data_path = pathlib.Path(__file__).parent()

    dataset_file_path = data_path / f"{dataset_name}.pkl"

    with open(dataset_file_path, "wb") as file_path:
        pickle.dump(dataset_raw, file_path)

    print(f'Saved dataset "{dataset_name}".')
    pass


def load_dataset_raw(dataset_name, data_path=None):
    if data_path is None:
        data_path = pathlib.Path(__file__).parent()

    dataset_file_path = data_path / f"{dataset_name}.pkl"

    with open(dataset_file_path, "rb") as file_path:
        dataset_raw = pickle.load(file_path)

    return dataset_raw
