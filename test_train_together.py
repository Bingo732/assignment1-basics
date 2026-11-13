if __name__ == "__main__":
    from cs336_basics.train_together import trainTogther

    train_src = "/Users/ovo/本地资源/学习/CS336/作业1/assignment1-basics/my_result/ts_train_np.npy"
    valid_src = "/Users/ovo/本地资源/学习/CS336/作业1/assignment1-basics/my_result/ts_valid_np.npy"
    out_dir = "/Users/ovo/本地资源/学习/CS336/作业1/assignment1-basics/my_result/"
    device = "mps"

    trainTogther(train_src, valid_src, out_dir, device)
