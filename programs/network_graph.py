import torch
import networkx as nx


# Function to extract graph based on activations from the network and a single input data point
def graph_DNN(model, input_data):
    G = nx.DiGraph()  # Directed graph
    activations = input_data.view(-1, 28 * 28)
    current_node = 0  # To keep track of node indices across layers

    _, outputs = model(input_data, return_intermediates=True)
    layer_weights = list(model.children())
    

    for ii, layer in enumerate(layer_weights):
        output = outputs[ii]
        
        input_dim, output_dim = layer.in_features, layer.out_features
        weight_matrix = layer.weight.detach()#.numpy()

        # # Set rows to zero where activated_nodes is False
        # weight_matrix[~activated_nodes] = 0

        edges = weight_matrix * output.detach()
        
        for i in range(input_dim):
            for j in range(output_dim):
                # Add edges based on the active neurons
                #if activated_nodes[j]:
                G.add_edge(current_node + i, current_node + input_dim + j, 
                            weight=abs(edges[j, i]))
        
        # Update current_node to next layer's node index start
        current_node += input_dim
    return G


# Function to extract graph based on activations from the network and a single input data point
def graph_DNN_class_based(model, data_batch, labels):

    model.eval()
    with torch.no_grad():
        activations = data_batch.view(-1, 28 * 28) # (activations == outputs[0]).all() TRUE!
        current_node = 0  # To keep track of node indices across layers

        _, outputs = model(data_batch, return_intermediates=True)
        layer_weights = list(model.children())

        G_list = [nx.DiGraph() for ii in labels.unique()]  # Directed graph
        for ii, layer in enumerate(layer_weights):

            output = outputs[ii]
            
            input_dim, output_dim = layer.in_features, layer.out_features
            weight_matrix = layer.weight.detach()#.numpy()

            # # Set rows to zero where activated_nodes is False
            # weight_matrix[~activated_nodes] = 0

            edges = weight_matrix.unsqueeze(0) * output.detach().unsqueeze(1) #torch.matmul(output.detach(), weight_matrix.T)

            for label in labels.unique():

                class_av_edge = edges[labels==label].mean(axis=0)

                for i in range(input_dim):
                    for j in range(output_dim):
                        # Add edges based on the active neurons
                        #if activated_nodes[j]:
                        G_list[label-1].add_edge(current_node + i, current_node + input_dim + j, 
                                    weight=abs(class_av_edge[j, i]))
                # Update current_node to next layer's node index start
                print(label)
                current_node += input_dim

    return G_list


# Function to extract graph based on activations from the network and a single input data point
def graph_DNN_layer_level(model, input_data):
    activations = input_data.view(-1, 28 * 28)
    current_node = 0  # To keep track of node indices across layers

    _, outputs = model(input_data, return_intermediates=True)
    layer_weights = list(model.children())

    G_list = []

    for ii, layer in enumerate(layer_weights):
        G = nx.DiGraph()  # Directed graph
        output = outputs[ii]
        
        input_dim, output_dim = layer.in_features, layer.out_features
        weight_matrix = layer.weight.detach()#.numpy()

        # # Set rows to zero where activated_nodes is False
        # weight_matrix[~activated_nodes] = 0

        edges = weight_matrix * output.detach()
        
        for i in range(input_dim):
            for j in range(output_dim):
                # Add edges based on the active neurons
                #if activated_nodes[j]:
                G.add_edge(current_node + i, current_node + input_dim + j, 
                            weight=abs(edges[j, i]))
        
        # Update current_node to next layer's node index start
        current_node += input_dim
        G_list.append(G)
        
    return G_list


def graph_DNN_class_based_selected_layers(model, data_batch, labels, selected_layers):

    model.eval()
    with torch.no_grad():
        activations = data_batch.view(-1, 28 * 28) # (activations == outputs[0]).all() TRUE!
        current_node = 0  # To keep track of node indices across layers

        _, outputs = model(data_batch, return_intermediates=True)
        layer_weights = list(model.children())

        G_list = [nx.DiGraph() for ii in labels.unique()]  # Directed graph
        for ii, layer in enumerate(layer_weights):

            if ii in selected_layers:
                output = outputs[ii]
                
                input_dim, output_dim = layer.in_features, layer.out_features
                weight_matrix = layer.weight.detach()#.numpy()

                # # Set rows to zero where activated_nodes is False
                # weight_matrix[~activated_nodes] = 0

                edges = weight_matrix.unsqueeze(0) * output.detach().unsqueeze(1) #torch.matmul(output.detach(), weight_matrix.T)
                for label in labels.unique():

                    class_av_edge = edges[labels==label].mean(axis=0).cpu()

                    for i in range(input_dim):
                        for j in range(output_dim):
                            # Add edges based on the active neurons
                            #if activated_nodes[j]:
                            G_list[label-1].add_edge(current_node + i, current_node + input_dim + j, 
                                        weight=abs(class_av_edge[j, i]))
                    # Update current_node to next layer's node index start
                    current_node += input_dim

    return G_list