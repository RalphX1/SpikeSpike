"""
Streamlit Web App for Spike1 Spiking Neural Network
Interactive visualization of LIF neurons and spike propagation
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from spike1 import SpikingNetwork

st.set_page_config(page_title="Spike1 - Spiking Neural Network", layout="wide")

# Title and description
st.title("🧠 Spike1: Spiking Neural Network Visualizer")
st.markdown("""
Interact with a simple Leaky Integrate-and-Fire (LIF) spiking neural network.
Adjust parameters and watch neurons spike in real-time!
""")

# Sidebar for controls
st.sidebar.header("⚙️ Network Configuration")

# Network topology selection
network_type = st.sidebar.selectbox(
    "Network Topology",
    ["Single Neuron", "Two Connected Neurons", "Small Network (4 neurons)"]
)

st.sidebar.header("🔧 Neuron Parameters")

# Neuron parameters
input_current = st.sidebar.slider(
    "Input Current (I)",
    min_value=0.0,
    max_value=30.0,
    value=15.0,
    step=0.5,
    help="External current applied to input neuron(s)"
)

tau_m = st.sidebar.slider(
    "Membrane Time Constant (τ_m)",
    min_value=5.0,
    max_value=20.0,
    value=10.0,
    step=1.0,
    help="How quickly the membrane potential decays"
)

v_thresh = st.sidebar.slider(
    "Threshold Potential (V_thresh)",
    min_value=-60.0,
    max_value=-40.0,
    value=-50.0,
    step=1.0,
    help="Voltage threshold for spiking"
)

v_reset = st.sidebar.slider(
    "Reset Potential (V_reset)",
    min_value=-80.0,
    max_value=-60.0,
    value=-70.0,
    step=1.0,
    help="Voltage after spike"
)

st.sidebar.header("⏱️ Simulation Parameters")

duration = st.sidebar.slider(
    "Simulation Duration (ms)",
    min_value=50,
    max_value=500,
    value=150,
    step=10
)

if network_type != "Single Neuron":
    synapse_weight = st.sidebar.slider(
        "Synaptic Weight",
        min_value=5.0,
        max_value=30.0,
        value=18.0,
        step=1.0,
        help="Strength of synaptic connections"
    )

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):

    with st.spinner("Running simulation..."):
        # Create network based on selected topology
        net = SpikingNetwork(dt=0.1)

        if network_type == "Single Neuron":
            net.add_neuron('n1', tau_m=tau_m, v_thresh=v_thresh, v_reset=v_reset)
            external_inputs = {'n1': input_current}
            neuron_ids = ['n1']

        elif network_type == "Two Connected Neurons":
            net.add_neuron('input', tau_m=tau_m, v_thresh=v_thresh, v_reset=v_reset)
            net.add_neuron('output', tau_m=tau_m, v_thresh=v_thresh, v_reset=v_reset)
            net.add_synapse('input', 'output', weight=synapse_weight)
            external_inputs = {'input': input_current}
            neuron_ids = ['input', 'output']

        else:  # Small Network
            for i in range(4):
                net.add_neuron(f'n{i}', tau_m=tau_m, v_thresh=v_thresh, v_reset=v_reset)
            net.add_synapse('n0', 'n1', weight=synapse_weight)
            net.add_synapse('n0', 'n2', weight=synapse_weight)
            net.add_synapse('n1', 'n3', weight=synapse_weight * 0.7)
            net.add_synapse('n2', 'n3', weight=synapse_weight * 0.7)
            external_inputs = {'n0': input_current}
            neuron_ids = ['n0', 'n1', 'n2', 'n3']

        # Run simulation
        results = net.run(
            duration=duration,
            external_inputs=external_inputs,
            record_v=True
        )

        # Store results in session state
        st.session_state['results'] = results
        st.session_state['neuron_ids'] = neuron_ids
        st.session_state['network_type'] = network_type

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    neuron_ids = st.session_state['neuron_ids']
    network_type = st.session_state['network_type']

    # Statistics
    st.header("📊 Simulation Results")

    cols = st.columns(len(neuron_ids))
    for idx, neuron_id in enumerate(neuron_ids):
        num_spikes = len(results['spikes'].get(neuron_id, []))
        with cols[idx]:
            st.metric(f"Neuron {neuron_id}", f"{num_spikes} spikes")

    # Visualization
    st.header("📈 Membrane Potential & Spikes")

    time = results['time_points']

    # Create figure with subplots
    num_neurons = len(neuron_ids)
    fig, axes = plt.subplots(num_neurons + 1, 1, figsize=(12, 2 + num_neurons * 1.5))

    if num_neurons == 1:
        axes = [axes[0], axes[1]]

    # Plot membrane potentials
    for idx, neuron_id in enumerate(neuron_ids):
        voltage = results['voltages'][neuron_id]
        axes[idx].plot(time, voltage, linewidth=1.5, color=f'C{idx}')
        axes[idx].set_ylabel(f'{neuron_id}\nV (mV)', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=v_thresh, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')

        if idx == 0:
            axes[idx].set_title(f'{network_type} - Membrane Potentials', fontsize=12, fontweight='bold')
            axes[idx].legend(loc='upper right', fontsize=8)

    # Spike raster plot
    spike_data = []
    colors = []
    color_palette = plt.cm.tab10(range(10))

    for idx, neuron_id in enumerate(neuron_ids):
        if neuron_id in results['spikes'] and results['spikes'][neuron_id]:
            spike_data.append(results['spikes'][neuron_id])
            colors.append(color_palette[idx])

    if spike_data:
        axes[-1].eventplot(spike_data, linewidths=2, colors=colors)
        axes[-1].set_yticks(range(len(spike_data)))
        axes[-1].set_yticklabels(neuron_ids[:len(spike_data)], fontsize=10)

    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    axes[-1].set_ylabel('Neuron', fontsize=10)
    axes[-1].set_title('Spike Raster Plot', fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, duration)
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Network diagram
    if network_type != "Single Neuron":
        st.header("🔗 Network Topology")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Connections:**")
            for synapse in net.synapses:
                st.text(f"{synapse.pre_neuron_id} → {synapse.post_neuron_id} (w={synapse.weight:.1f})")

        with col2:
            # Simple network diagram
            fig_net, ax = plt.subplots(figsize=(8, 6))

            if network_type == "Two Connected Neurons":
                positions = {'input': (0, 0), 'output': (2, 0)}
            else:  # Small Network
                positions = {
                    'n0': (0, 1),
                    'n1': (2, 2),
                    'n2': (2, 0),
                    'n3': (4, 1)
                }

            # Draw neurons
            for neuron_id, (x, y) in positions.items():
                num_spikes = len(results['spikes'].get(neuron_id, []))
                color = 'red' if num_spikes > 0 else 'lightblue'
                ax.scatter(x, y, s=2000, c=color, alpha=0.7, edgecolors='black', linewidths=2)
                ax.text(x, y, f'{neuron_id}\n({num_spikes})', ha='center', va='center',
                       fontsize=12, fontweight='bold')

            # Draw connections
            for synapse in net.synapses:
                x1, y1 = positions[synapse.pre_neuron_id]
                x2, y2 = positions[synapse.post_neuron_id]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))

            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 2.5)
            ax.axis('off')
            ax.set_title('Network Structure (number = spike count)', fontsize=12, fontweight='bold')

            st.pyplot(fig_net)

else:
    # Show instructions
    st.info("👈 Configure the network parameters in the sidebar and click **Run Simulation** to start!")

    # Show example
    st.header("ℹ️ About This App")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Features
        - **Interactive Parameters**: Adjust neuron properties in real-time
        - **Multiple Topologies**: Single neuron, connected pairs, or small networks
        - **Real-time Visualization**: See membrane potentials and spike trains
        - **Network Diagrams**: Visualize connections and activity

        ### Neuron Model
        This app uses the **Leaky Integrate-and-Fire (LIF)** model:

        ```
        dV/dt = -(V - V_rest)/τ_m + I/C
        ```

        When V ≥ V_thresh, the neuron spikes and V resets to V_reset.
        """)

    with col2:
        st.markdown("""
        ### How to Use
        1. Select a network topology from the sidebar
        2. Adjust neuron parameters (input current, threshold, etc.)
        3. Set simulation duration
        4. Click **Run Simulation**
        5. Explore the results!

        ### Parameters Guide
        - **Input Current**: Higher values → more spikes
        - **τ_m**: Higher values → slower response
        - **V_thresh**: Lower values → easier to spike
        - **Synaptic Weight**: Stronger connections between neurons
        """)

# Footer
st.markdown("---")
st.markdown("Built with Spike1 - Simple Spiking Neural Network | [GitHub](https://github.com/RalphX1/SpikeSpike)")
