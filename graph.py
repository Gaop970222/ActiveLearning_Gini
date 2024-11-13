import matplotlib.pyplot as plt
# Redefine the data and colors due to system reset
data = {
    "IF = 0.3": {
        "Random Sampling": [0.8106, 0.8225, 0.8449, 0.8587],
        "Nearest": [0.8154, 0.8336, 0.8466, 0.8594],
        "BALDDropout": [0.8213, 0.8471, 0.8539, 0.868], 
        "Entropy Sampling": [0.8352, 0.8544, 0.8832, 0.8968],
        "Probability imbalance": [0.8329, 0.8591, 0.8876, 0.8991],
        "Gini": [0.8272, 0.8613, 0.8857, 0.8964],
        "Gini 0.5": [0.8378, 0.8625, 0.8809, 0.898]
    },
    "IF = 0.1": {
        "Random Sampling": [0.8093, 0.8222, 0.8455, 0.8506],
        "Nearest": [0.8208, 0.8425, 0.8596, 0.8583],
        "BALDDropout": [0.8246, 0.8371, 0.8607, 0.8682],
        "Entropy Sampling": [0.8263, 0.8455, 0.8571, 0.8692],
        "Probability Imbalance": [0.8293, 0.8441, 0.8629, 0.8662],
        "Gini": [0.8213, 0.8447, 0.8624, 0.8728],
        "Gini 0.5": [0.8182, 0.8456, 0.8609, 0.8762]
    },
    "IF = 0.5": {
        "Random Sampling": [0.8223, 0.8413, 0.8584, 0.8753],
        "Nearest": [0.8305, 0.8526, 0.8697, 0.8835],
        "BALDDropout": [0.837, 0.868, 0.8852, 0.901],
        "Entropy Sampling": [0.8251, 0.8563, 0.8908, 0.8982],
        "Probability": [0.8331, 0.8574, 0.8814, 0.8959],
        "Gini 0.5": [0.8304, 0.8641, 0.886, 0.907],
        "Gini": [0.8354, 0.863, 0.8873, 0.902]
    }
}

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'black']

# Now let's create separate plots for each imbalance factor
# Save the plots to files
for i, (imbalance_factor, methods) in enumerate(data.items()):
    fig, ax = plt.subplots(figsize=(8, 6))
    for j, (method, values) in enumerate(methods.items()):
        ax.plot(range(1, len(values) + 1), values, label=method, color=colors[j], linewidth=2.5)
    ax.set_title(f"Performance for {imbalance_factor}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Value")
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True)

    # Save the plot to a file
    plt.savefig(f"Performance_for_{imbalance_factor.replace(' ', '_').replace('=', '')}.png")

