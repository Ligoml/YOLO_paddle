import numpy as np


def gen_fake_data():
    fake_data = np.random.rand(32, 3, 416, 416).astype(np.float32) - 0.5
    fake_label = np.random.rand(32, 2, 5).astype(np.float32)
    fake_label_1 = np.random.randint(20, size=32).astype(np.float32)
    fake_label_2 = np.random.randint(20, size=32).astype(np.float32)
    for i in range(32):
        fake_label[i, 0, 4] = fake_label_1[i]
        fake_label[i, 1, 4] = fake_label_2[i]

    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()