from ..utils.data_loader import create_dataloader
def test_data_pipeline():
    

    max_len = 8  # Example max_len
    batch_size = 2

    dataloader = create_dataloader("./", max_len=max_len, batch_size=batch_size)

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Input IDs:", input_ids)
        print("Target IDs:", target_ids)
        break

# Run the test
test_data_pipeline()