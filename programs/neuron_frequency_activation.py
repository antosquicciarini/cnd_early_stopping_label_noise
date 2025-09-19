import torch
import networkx as nx
import matplotlib.pyplot as plt


def neuron_frequency_activation(loader, loader_corrupted, loader_not_corrupted, model, corrupted_samples, device, args):
    #corrupted_samples = corrupted_samples[corrupted_samples<args.fixed_batch_size]
    FA_per_neuron = None
    FA_per_instance = None

    model.eval()
    with torch.no_grad():

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images, return_intermediates=True, neuron_indexes=slice(None))
            if FA_per_neuron == None:
                FA_per_neuron = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0) 
                FA_per_instance = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]
            else:
                FA_per_neuron = FA_per_neuron + torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0)
                FA_per_instance = torch.cat([FA_per_instance,  torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]]) 
        FA_per_neuron = FA_per_neuron/len(FA_per_instance)

        corrupted_samples = torch.sort(torch.tensor(corrupted_samples))[0]
        mask = torch.tensor([i in corrupted_samples for i in range(len(FA_per_instance))])

        # FA_per_neuron_cor = None
        # FA_per_instance_cor = None

        # for images, labels in loader_corrupted:
        #     images, labels = images.to(device), labels.to(device)
        #     _, outputs = model(images, return_intermediates=True, neuron_indexes=slice(None))
        #     if FA_per_neuron_cor == None:
        #         FA_per_neuron_cor = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0) 
        #         FA_per_instance_cor = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]
        #     else:
        #         FA_per_neuron_cor = FA_per_neuron_cor + torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0)
        #         FA_per_instance_cor = torch.cat([FA_per_instance_cor,  torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]]) 
        # FA_per_neuron_cor = FA_per_neuron_cor/len(FA_per_instance_cor)



        # FA_per_neuron_no_cor = None
        # FA_per_instance_no_cor = None

        # for images, labels in loader_not_corrupted:
        #     images, labels = images.to(device), labels.to(device)
        #     _, outputs = model(images, return_intermediates=True, neuron_indexes=slice(None))
        #     if FA_per_neuron_no_cor == None:
        #         FA_per_neuron_no_cor = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0) 
        #         FA_per_instance_no_cor = torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]
        #     else:
        #         FA_per_neuron_no_cor = FA_per_neuron_no_cor + torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=0)
        #         FA_per_instance_no_cor = torch.cat([FA_per_instance_no_cor,  torch.where(outputs != 0, torch.tensor(1), outputs).sum(axis=1) / outputs.shape[1]]) 
        # FA_per_neuron_no_cor = FA_per_neuron_no_cor/len(FA_per_instance_no_cor)

    

        # plt.plot(torch.linspace(0, 1, len(FA_per_neuron_no_cor)), torch.sort(FA_per_neuron_no_cor.cpu())[0], linewidth=3, alpha=0.5, label="good data")
        # plt.plot(torch.linspace(0, 1, len(FA_per_neuron_cor)), torch.sort(FA_per_neuron_cor.cpu())[0], linewidth=1, alpha=0.8, label="polluted data")
        # plt.title(f"Neuron activation rate - Epoch: {args.current_epoch}")
        # plt.tight_layout()
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"/Users/antoniosquicciarini/Desktop/neuron_act_experiment/xNeuron_epoch_{args.current_epoch}.png")
        # plt.close()

        # plt.plot(torch.linspace(0, 1, len(FA_per_instance_no_cor)), torch.sort(FA_per_instance_no_cor.cpu())[0], linewidth=3, alpha=0.5, label="good data")
        # plt.plot(torch.linspace(0, 1, len(FA_per_instance_cor)), torch.sort(FA_per_instance_cor.cpu())[0], linewidth=1, alpha=0.8, label="polluted data")
        # plt.title(f"Instance activation rate - Epoch: {args.current_epoch}")
        # plt.tight_layout()
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"/Users/antoniosquicciarini/Desktop/neuron_act_experiment/xInstance_epoch_{args.current_epoch}.png")
        # plt.close()

        
        return FA_per_neuron, FA_per_instance

# plt.plot(torch.linspace(0, 1, len(FA_per_neuron)), torch.sort(FA_per_neuron.cpu())[0])
# plt.plot(torch.linspace(0, 1, len(FA_per_neuron)-corrupted_samples), torch.sort(FA_per_neuron[~mask].cpu())[0])
# plt.plot(torch.linspace(0, 1, len(FA_per_neuron)), torch.sort(FA_per_neuron.cpu())[0])

# plt.plot(torch.linspace(0, 1, len(FA_per_instance)), torch.sort(FA_per_instance.cpu())[0], linewidth=2, alpha=0.5, label="all data")
# plt.plot(torch.linspace(0, 1, len(FA_per_instance[mask])), torch.sort(FA_per_instance[mask].cpu())[0], linewidth=1, alpha=0.5, label="good data")
# plt.plot(torch.linspace(0, 1, len(FA_per_instance[~mask])), torch.sort(FA_per_instance[~mask].cpu())[0], linewidth=1, alpha=0.5, label="polluted data")
# plt.title(f"Epoch: {args.current_epoch}")
# plt.tight_layout()
# plt.legend()
# plt.grid()

def neuron_frequency_activation_plot(frequency_activation_list, performances):

    return None

# sorted_tensor, indices = torch.sort(frequency_activation_list[0][1])
# plt.plot(sorted_tensor.detach().cpu())