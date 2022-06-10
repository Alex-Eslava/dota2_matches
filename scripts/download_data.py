
import time
import kaggle


def main():
    print("Starting download...")
    start_time = time.time()
    
    dataset_name = 'devinanzelmo/dota-2-matches'
    data_path = '../data/external/dota2_dataset'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name, path=data_path, unzip=True)

    print(f"Download finished in {round((time.time()-start_time)/60, 2)} minutes")


if __name__ == "__main__":
    main()