from Train_Test import train, test, parameter_selection

'''
Please kindly cite this paper 
Dunbin Shen, Xiaorui Ma, Wenfeng Kong, Jianjun Liu, Jie Wang, Hongyu Wang. 
"Hyperspectral Target Detection Based on Interpretable Representation Network," 
IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3302950.
'''


def main(model_config=None):
    modelConfig = {
        "state": "test",  # or train, or parameter_selection
        "dataset": "Sandiego",  # dataset
        "m": 30,  # the number of background endmembers which can be tuned
        "eta1": 10,  # the regularization parameter of sparse loss which can be tuned
        "lr": 1e-3,  # the initial learning rate
        "weight_decay": 2e-5,  # the weight-decay parameter
        "iteration": 5000,  # the total number of training iterations
        "model_path": "model/",  # the path of the saved model
        "epision": 4,  # the number of decimal places of the quantitative results
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "test":
        test(modelConfig)
    else:
        parameter_selection(modelConfig)


if __name__ == '__main__':
    main()
