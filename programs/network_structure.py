import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import ExampleTiedDropout
import torchvision
import torchvision.models as models
from net_FCN import FullyConnectedNN, FullyConnectedNN_masked, FullyConnectedNN_preact_CND, NewsMLP
from net_resnet import ResNet, ResNet_Fast, BasicBlock, Bottleneck, ResNet50p, ResNet9_dropout
from net_preact_resnet import PreActResNet, PreActBlock
from net_VGG19 import VGG19
from net_LeNet import LeNet
from net_ViT import LightweightViT
import matplotlib.pyplot as plt

def model_definition(device, args, embedding_weights=None):
    if args.network == "Base_FCN":
        model = FullyConnectedNN(device, 
                                 input_channels = args.input_channels, 
                                 input_size = args.input_size, 
                                 N = args.n_neurons_x_layer,
                                 L = args.n_layers,
                                 batch_normalization_flag=getattr(args, "batch_normalization_flag", False),
                                 dropout_rate=getattr(args, "dropout_rate", 0),
                                 ).to(device)
        neurs_x_hid_lyr = {i: args.n_neurons_x_layer for i in range(args.n_layers)}

    elif args.network == "Preact_Base_FCN":
        model = FullyConnectedNN_preact_CND(device, 
                                 input_channels = args.input_channels, 
                                 input_size = args.input_size, 
                                 num_classes=args.num_classes,
                                 N = args.n_neurons_x_layer,
                                 L = args.n_layers,
                                 batch_normalization_flag=getattr(args, "batch_normalization_flag", False),
                                 dropout_rate=getattr(args, "dropout_rate", 0),
                                 embedding_weights = embedding_weights,
                                 activation_fn=getattr(args, "activation_fn", 'relu'),
                                 ).to(device)
        neurs_x_hid_lyr = {i: args.n_neurons_x_layer for i in range(args.n_layers)}

    elif args.network == "NewsMLP":
        # Fixed no modular class designed for NEWS basen on Yu 19 article
        model = NewsMLP(device, 
                        embedding_weights,
                        num_classes = args.num_classes,
                        dropout_rate=args.dropout,
                        ).to(device)
        neurs_x_hid_lyr = {
            0: 300*20,
            1: 300*4,
            2: 300
        }

    elif args.network == "Base_FCN_masked":
        model = FullyConnectedNN_masked(device).to(device)


    elif args.network == "VGG19":
        model = VGG19(input_channels=args.input_channels, num_classes=args.num_classes).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512,
            4: 512,
            5: 4096,
            6: 4096
        }

    elif args.network == "LeNet":
        model = LeNet(input_channels=args.input_channels, num_classes=args.num_classes, dropout_rate=getattr(args, "dropout_rate", 0.0)).to(device)
        neurs_x_hid_lyr = {
            0: 20,
            1: 50,
            2: 500
        }


    elif "ResNet9" == args.network:
        model = ResNet(BasicBlock, [1, 1, 1, 1],  input_channels=args.input_channels, num_classes=args.num_classes, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }
    elif "ResNet18" == args.network:
        model = ResNet(BasicBlock, [2, 2, 2, 2],  input_channels=args.input_channels, num_classes=args.num_classes, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }

    elif "ResNet34" == args.network:
        # ResNet-34: 3, 4, 6, 3 layers in each block
        model =  ResNet(BasicBlock, [3, 4, 6, 3], num_classes=args.num_classes, input_channels=args.input_channels, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }

    elif "ResNet50" == args.network:
        # ResNet-50: 3, 4, 6, 3 layers in each block using Bottleneck block
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=args.num_classes, input_channels=args.input_channels, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 256,
            1: 512,
            2: 1024,
            3: 2048
        }



    elif args.network  == "ResNet34_Fast":
        # ResNet-34: 3, 4, 6, 3 layers in each block
        model =  ResNet_Fast(args).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }

    elif args.network  == "ResNet50_Fast":
        # ResNet-50: 3, 4, 6, 3 layers in each block using Bottleneck block
        model = ResNet_Fast(args).to(device)
        neurs_x_hid_lyr = {
            0: 256,
            1: 512,
            2: 1024,
            3: 2048
        }




    elif args.network  == "ResNet50p":
        model = ResNet50p(num_classes=args.num_classes).to(device)
        neurs_x_hid_lyr = {
            0: 256,
            1: 512,
            2: 1024,
            3: 2048
        }
    elif args.network == "ResNet9_Maini":
        model = ResNet9_dropout(device, p_fixed = args.p_fixed, p_mem = args.p_mem, num_classes=args.num_classes, input_channels=args.input_channels).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 128,
            3: 256,
            4: 256,
            5: 128
        }




    elif "PreAct_ResNet9" == args.network:
        model = PreActResNet(PreActBlock, [1, 1, 1, 1],  input_channels=args.input_channels, num_classes=args.num_classes, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }
    elif "PreAct_ResNet18" == args.network:
        model = PreActResNet(PreActBlock, [2, 2, 2, 2],  input_channels=args.input_channels, num_classes=args.num_classes, dropout_rate=getattr(args, "dropout_rate", 0.0), network=args.network).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 128,
            2: 256,
            3: 512
        }


    elif args.network == "LightweightViT":
        model = LightweightViT(num_classes=args.num_classes, input_channels=args.input_channels).to(device)
        neurs_x_hid_lyr = {
            0: 64,
            1: 64,
            2: 64,
            3: 64
        }

    else:
        raise ValueError(f"Unsupported model_type: {args.network}")
    print(f"Model Selected {args.network}")


    por_neuron = getattr(args, "por_neurons_x_layer_preact", 1.)
    args.neurs_x_hid_lyr = {key: int(value * por_neuron) for key, value in neurs_x_hid_lyr.items()}

    if getattr(args, "n_GPUs", 1) > 1:
        print("PARALLEL MODE ACTIVATED")
        model = torch.nn.DataParallel(model)

    return model, args