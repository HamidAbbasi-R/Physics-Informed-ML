import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
st.set_page_config(layout="wide")

# Title and description
st.title("Van der Pol Oscillator with Physics-Informed Tensor Basis Neural Network (TBNN)")
st.write("""
    This example demonstrates how to train a TBNN to learn the dynamics of the Van der Pol oscillator.
    The Van der Pol oscillator is a nonlinear oscillator with damping. 
    The dynamics of the Van der Pol oscillator are governed by the following ordinary differential equations (ODEs):
         
    $$ \\frac{dx}{dt} = y $$
         
    $$ \\frac{dy}{dt} = \\mu (1 - x^2) y - x $$
         
    where $x$ and $y$ are the state variables, $\\mu$ is the damping parameter, and $t$ is time.
    The goal is to train a TBNN to predict the state variables $x(t)$ and $y(t)$ for a given initial condition.
    The TBNN is trained using both the data loss (mean squared error) and the physics loss (residuals of the ODEs).
    For training, we generate random initial conditions and solve the ODEs numerically to obtain the true solutions.
         
    $$ \\text{Initial Conditions}: \\quad x(0) = rand(-2, 2), \\quad y(0) = rand(-2, 2) $$

    The TBNN is trained to minimize the total loss, which is a combination of the data loss and the physics loss.
    After training, we evaluate the TBNN on new initial conditions and compare the predicted trajectories with the true solutions.
    The results are visualized in the phase space and time series plots.
    """)

st.write("""
    First set the parameters for the Van der Pol oscillator and the TBNN training.
    You can adjust the parameters as needed.
    But keep in mind that for large number of hidden units in the TBNN, large number of training epochs, or large number of initial conditions for training, the model may take longer to train.
    The default parameters should work well for this example.
    """)

st.write("""
    Next, we design the TBNN architecture using PyTorch.
    The TBNN consists of three fully connected layers with ReLU activation functions.
    The input dimension is 2 (initial conditions), and the output dimension is 2 (predicted variables x(t) and y(t)) times the number of time steps.
    The number of hidden units in the TBNN is a hyperparameter that can be adjusted.
    The TBNN is trained using the Adam optimizer and the mean squared error loss and the physics loss.
    The physics loss is the mean squared error of the residuals of the Van der Pol equations.
    The TBNN is trained to minimize the total loss, which is a combination of the data loss and the physics loss.
    Physical loss $$L_P$$ is calculated using the following equations:
        
    $$ L_P = \\frac{1}{N} \\sum_{i=1}^{N} \\left(\\frac{dx}{dt} - y\\right)^2 + \\left(\\frac{dy}{dt} - (\\mu (1 - x^2) y - x)\\right)^2 $$
        
    When you first load the app, the TBNN will be trained with the default parameters.
    Every time you submit the form with new parameters, the TBNN will be retrained with the new parameters.
    Make sure the app is not loading when you see the new results.
    The loading icon on the top right corner will disappear when the app is ready.
    """) 

# Initialize session state
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "tbnn_model" not in st.session_state:
    st.session_state.tbnn_model = None

# Parameters for generating solutions
st.sidebar.header("Training Parameters")
t_end = 10
N_time_steps = 200
num_samples = st.sidebar.slider("Number of initial conditions for training", 10, 500, 100, step=10)
hidden_dim = st.sidebar.slider("Number of hidden units in the TBNN", 16, 128, 64, step=16)
num_epochs = st.sidebar.slider("Number of training epochs", 500, 5000, 500, step=100)

# Button to trigger training
train_button = st.sidebar.button("Train TBNN")

st.sidebar.header("Test Parameters for Evaluation")
mu_0 = st.sidebar.slider("Van der Pol oscillator parameter, $\\mu$", 0.5, 1.5, 1.0, step=0.1)
x0 = st.sidebar.slider("Initial condition x0", -2.0, 2.0, -1.0, step=0.1)
y0 = st.sidebar.slider("Initial condition y0", -2.0, 2.0, -1.5, step=0.1)

# Define the Van der Pol system
def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# Cache the data generation process
@st.cache_data
# Generate training data
def generate_training_data(t_end, N_time_steps, num_samples):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, N_time_steps)

    # Generate random initial conditions
    np.random.seed(42)
    initial_conditions = np.random.uniform(-2, 2, size=(num_samples, 2))
    mu = np.random.uniform(0.5, 1.5, size=(num_samples, 1))

    # Solve the ODE for each initial condition
    inputs = []
    outputs = []

    for ic,mu in zip(initial_conditions, mu):
        sol = solve_ivp(van_der_pol, t_span, ic, args=(mu), t_eval=t_eval, method='RK45')
        solution = sol.y.T  # Shape: (time_steps, 2)
        for t in range(len(t_eval) - 1):
            inputs.append([solution[t, 0], solution[t, 1], mu[0]])  # Current state (x_t, y_t, mu)
            outputs.append(solution[t + 1])  # Next state (x_{t+1}, y_{t+1})


    inputs = np.array(inputs)  # Shape: (num_samples * (N_time_steps - 1), 3)
    outputs = np.array(outputs)  # Shape: (num_samples * (N_time_steps - 1), 2)
    return inputs, outputs, t_eval

# Generate training data
inputs, outputs, t_eval = generate_training_data(t_end, N_time_steps, num_samples)

# preprocess the data using MinMaxScaler
scaler = MinMaxScaler()
scaler_mu = MinMaxScaler()
inputs_xy = scaler.fit_transform(inputs[:, :2])
scaled_mu = scaler_mu.fit_transform(inputs[:, 2].reshape(-1, 1))
inputs = np.concatenate((inputs_xy, scaled_mu), axis=1)
outputs = scaler.fit_transform(outputs)


# Define the TBNN architecture
class TBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TBNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Predicts (x_{t+1}, y_{t+1})
        return x

# Train the model if the button is clicked
if train_button or not st.session_state.model_trained:
    st.write("Training the TBNN...")

    # Hyperparameters
    input_dim = 3  # (x_t, y_t, mu)
    output_dim = 2  # (x_{t+1}, y_{t+1})

    # Instantiate the model
    model = TBNN(input_dim, hidden_dim, output_dim)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    # Define the physics loss
    def physics_loss(predictions, inputs):
        # inverse transform the inputs
        inputs_xy = scaler.inverse_transform(inputs[:, :2])
        input_mu = scaler_mu.inverse_transform(inputs[:, 2].reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions.detach().numpy())

        x_pred, y_pred = predictions[:, 0], predictions[:, 1]  # Predicted (x_{t+1}, y_{t+1})
        x_t, y_t = inputs_xy[:, 0], inputs_xy[:, 1]  # Current state (x_t, y_t)
        mu = input_mu[0][0]

        # Compute derivatives using finite differences
        dt = t_eval[1] - t_eval[0]
        dxdt = (x_pred - x_t) / dt
        dydt = (y_pred - y_t) / dt

        # Compute residuals of the Van der Pol equations
        res_x = dxdt - y_t
        res_x = torch.tensor(res_x, dtype=torch.float32)
        res_y = dydt - (mu * (1 - x_t**2) * y_t - x_t)
        res_y = torch.tensor(res_y, dtype=torch.float32)

        # Physics loss is the mean squared error of the residuals
        return torch.mean(torch.square(res_x) + torch.square(res_y))

    # Training loop
    loss_history = []
    phy_loss_history = []
    data_loss_history = []
    epoch_history = []

    progress_bar = st.progress(0)
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs_tensor)

        # Compute losses
        data_loss = mse_loss(predictions, outputs_tensor)
        phy_loss = physics_loss(predictions, inputs_tensor)
        total_loss = data_loss + phy_loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Store loss and epoch for plotting
        phy_loss_history.append(phy_loss.item())
        data_loss_history.append(data_loss.item())
        loss_history.append(total_loss.item())
        epoch_history.append(epoch)

        if epoch % 10 == 0:
            progress_bar.progress(epoch / num_epochs)

    progress_bar.progress(1.0)
    st.success("Training completed!")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch_history, y=phy_loss_history, mode='lines', name='Physics Loss', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=epoch_history, y=data_loss_history, mode='lines', name='Data Loss', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=epoch_history, y=loss_history, mode='lines', name='Total Loss', line=dict(color='blue', width=2)))
    fig.update_layout(title="Training Loss Curve", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white", yaxis_type="log")
    st.plotly_chart(fig)
    # Save the trained model in session state
    st.session_state.tbnn_model = model
    st.session_state.model_trained = True
    st.write("Training completed!")

# Evaluate the model
st.header("Evaluate the Model")


if st.session_state.model_trained:
    st.write("""
        Model has been trained. You can now evaluate the model.
        Change the initial conditions in the sidebar to see the predicted trajectories.
        """)
    model = st.session_state.tbnn_model

    # Test on new initial conditions
    test_initial_conditions = np.array([[x0, y0, mu_0]])

    # scale the initial conditions
    test_initial_conditions_xy = scaler.transform(test_initial_conditions[:, :2])
    predicted_trajectory = [test_initial_conditions_xy[0]]
    mu_scaled = scaler_mu.transform(test_initial_conditions[:, 2].reshape(-1, 1))
    test_initial_conditions = np.concatenate((test_initial_conditions_xy, mu_scaled), axis=1)

    # Predict solutions
    test_predictions = np.zeros((1, N_time_steps, 2))
    with torch.no_grad():
            current_state = torch.tensor(test_initial_conditions, dtype=torch.float32)
            for i in range(N_time_steps - 1):
                next_state = model(current_state).numpy()[0]  # Predict next state
                test_predictions[0, i] = next_state
                predicted_trajectory.append(next_state)
                next_state_array = np.array([[next_state[0], next_state[1], mu_scaled[0][0]]])
                current_state = torch.tensor(next_state_array, dtype=torch.float32)
        
            test_predictions[0] = np.array(predicted_trajectory)


    # inverse transform the test_initial_conditions
    test_initial_conditions = scaler.inverse_transform(test_initial_conditions[:, :2])
    sol = solve_ivp(van_der_pol, (0, t_end), test_initial_conditions[0], args=([mu_0]), t_eval=t_eval, method='RK45')
    true_solutions = sol.y.T  # Shape: (time_steps, 2)

    # Calculate MSE
    # inverse transform the test_predictions
    test_predictions = scaler.inverse_transform(test_predictions[0])
    test_pred_x = test_predictions[:, 0]
    test_pred_y = test_predictions[:, 1]

    MSE_x = (true_solutions[:, 0] - test_pred_x)**2
    MSE_y = (true_solutions[:, 1] - test_pred_y)**2

    # Plot results
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=t_eval, y=true_solutions[:, 0], mode='lines', name="True Solution"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=test_pred_x, mode='lines', name="TBNN Prediction"), row=1, col=1)
    fig.update_layout(yaxis_title="x(t)", xaxis1=dict(showticklabels=False, showgrid=False))

    fig.add_trace(go.Scatter(x=t_eval, y=true_solutions[:, 1], mode='lines', name="True Solution"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=test_pred_y, mode='lines', name="TBNN Prediction"), row=2, col=1)
    fig.update_layout(xaxis2_title="Time t", yaxis2_title="y(t)")
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_solutions[:, 0], y=true_solutions[:, 1], mode='lines', name="True Solution"))
    fig.add_trace(go.Scatter(x=test_pred_x, y=test_pred_y, mode='lines', name="TBNN Prediction"))
    fig.update_layout(xaxis_title="x(t)", yaxis_title="y(t)", xaxis=dict(scaleanchor="y", scaleratio=1))
    st.plotly_chart(fig)

    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=MSE_x, name="x(t) MSE", opacity = 0.5))
    # fig.add_trace(go.Histogram(x=MSE_y, name="y(t) MSE", opacity = 0.5))
    # fig.update_layout(barmode='overlay', xaxis_title="MSE", yaxis_title="Count")
    # st.plotly_chart(fig)
else:
    st.write("Please train the model first.")