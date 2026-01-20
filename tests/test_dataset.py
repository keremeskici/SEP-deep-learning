from data.dataset import FER_Dataset

def main():
    ds = FER_Dataset(root_dir="data/training", set_type="train")

    print("num_samples:", len(ds))
    print("classes:", ds.classes_list)
    print("class_dict:", ds.class_dict)

    x, y = ds[0]
    print("x type:", type(x))
    print("x shape:", x.shape)        # erwartet: torch.Size([3, 64, 64])
    print("y:", y, "dtype:", y.dtype) # erwartet: torch.int64

if __name__ == "__main__":
    main()
