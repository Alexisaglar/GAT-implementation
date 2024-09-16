import argparse
from os import write
from pickle import BININT
from threading import ExceptHookArgs
import time

from networkx import nodes
import torch
import torch.nn as nn
from torch.optim import Adam

from data.prepare_data import load_graph_data
from utils import utils_functions
from utils.constants import * 
from models.GAT import GAT


def get_main_loop(
    config,
    gat,
    cross_entropy_loss,
    optimiser,
    node_features,
    node_labels,
    edge_index,
    train_indices,
    val_indices,
    test_indices,
    patience_period,
    time_start,
):
    node_dim = 0 #node axis

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)
    
    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index) # data is packed into tuples because GAT uses nn.sequential which requires it that way

    def get_node_labels_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices, train_labels
        elif phase == LoopPhase.VAL:
            return val_indices, val_labels
        else:
            return test_indices, test_labels

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices, gt_node_labels = get_node_labels_indices(phase)

        # shape = (N, C) where N is the number of nodes and C is the number of classes
        nodes_unnormalised_scores = gat(graph_data)[0].index_select(node_dim, node_indices)
        
        # get every class and apply softmax and then calculate the cross entropy loss.
        loss = cross_entropy_loss(nodes_unnormalised_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimiser.zero_grad() # clean the trainable weights gradients in the computational graph
            loss.backward() # compute the gradients for every trainable weight
            optimiser.step() # apply the gradients to weights

        # Calculate the main metric - accuracy
        class_predictions = torch.argmax(nodes_unnormalised_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        # Logging
        if phase == LoopPhase.TRAIN:
            if config['enable_tensorboard']:
                pass
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["datasetname"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            if config['enable_tensorboard']:
                pass

            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            if accuracy > BEST_VAL_PERF or loss.item < BEST_VAL_LOSS:
                BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)
                BEST_VAL_LOSS = min(loss.item, BEST_VAL_LOSS)
                PATIENCE_CNT = 0 
            else:
                PATIENCE_CNT += 1

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping trainning, there is no more patience for this training') 
        else:
            return accuracy
        
    return(main_loop)

def train_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(config, device)

    # model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False,
    ).to(device)

    # training utilities
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimiser = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimiser,
        node_features,
        node_labels,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
        config['patience_period'],
        time.time()
    )

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]

    for epoch in range(config['num_of_epochs']):
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e: # patience has run out exception :0
                print(str(e))
                break


    if config['should_test']:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_acc
        print(f'Test accuracy = {test_acc}')
    else:
        config['test_perf'] = -1

    torch.save(
        utils_functions.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, utils_functions.get_available_binary_name(config['dataset_name']))

    )

def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualise", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':
    train_gat_cora(get_training_args())

