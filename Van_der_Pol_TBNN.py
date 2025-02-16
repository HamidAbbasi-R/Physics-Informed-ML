import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

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
mu = st.sidebar.slider("Van der Pol oscillator parameter, $\\mu$", 0.1, 5.0, 1.0, step=0.1)
t_end = 10
N_time_steps = 100
num_samples = st.sidebar.slider("Number of initial conditions for training", 100, 2000, 1000, step=100)
hidden_dim = st.sidebar.slider("Number of hidden units in the TBNN", 16, 128, 64, step=16)
num_epochs = st.sidebar.slider("Number of training epochs", 500, 5000, 500, step=100)

# Button to trigger training
train_button = st.sidebar.button("Train TBNN")

st.sidebar.header("Test Parameters for Evaluation")
x0 = st.sidebar.slider("Initial condition x0", -2.0, 2.0, -1.0, step=0.1)
y0 = st.sidebar.slider("Initial condition y0", -2.0, 2.0, -1.5, step=0.1)

# Cache the data generation process
@st.cache_data
def generate_training_data(mu, t_end, N_time_steps, num_samples):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, N_time_steps)

    # Define the Van der Pol system
    def van_der_pol(t, z):
        x, y = z
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        return [dxdt, dydt]

    # Generate random initial conditions
    np.random.seed(42)
    initial_conditions = np.random.uniform(-2, 2, size=(num_samples, 2))

    # Solve the ODE for each initial condition
    solutions = []
    for ic in initial_conditions:
        sol = solve_ivp(van_der_pol, t_span, ic, t_eval=t_eval, method='RK45')
        solutions.append(sol.y.T)  # Store the solution as (time_steps, 2)

    solutions = np.array(solutions)  # Shape: (num_samples, time_steps, 2)
    return initial_conditions, solutions, t_eval

# Generate training data if needed
initial_conditions, solutions, t_eval = generate_training_data(mu, t_end, N_time_steps, num_samples)

# Define the TBNN architecture
class TBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_time_steps):
        super(TBNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * num_time_steps)
        self.num_time_steps = num_time_steps
        self.output_dim = output_dim

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.num_time_steps, self.output_dim)
        return x

# Train the model if the button is clicked
if train_button or not st.session_state.model_trained:
    st.write("Training the TBNN...")

    # Hyperparameters
    input_dim = 2
    output_dim = 2
    num_time_steps = len(t_eval)

    # Instantiate the model
    model = TBNN(input_dim, hidden_dim, output_dim, num_time_steps)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Convert data to PyTorch tensors
    initial_conditions_tensor = torch.tensor(initial_conditions, dtype=torch.float32)
    solutions_tensor = torch.tensor(solutions, dtype=torch.float32)

    # Define the physics loss
    def physics_loss(predictions, mu=mu):
        batch_size, num_time_steps, _ = predictions.shape
        x_pred = predictions[:, :, 0]
        y_pred = predictions[:, :, 1]

        dt = t_eval[1] - t_eval[0]
        dxdt = torch.gradient(x_pred, dim=1)[0] / dt
        dydt = torch.gradient(y_pred, dim=1)[0] / dt

        residual_x = dxdt - y_pred
        residual_y = dydt - (mu * (1 - x_pred**2) * y_pred - x_pred)
        return torch.mean(residual_x**2 + residual_y**2)

    # Training loop
    loss_history = []  # To store loss values for plotting
    phy_loss_ihstory = []
    data_loss_history = []
    epoch_history = []  # To store epoch numbers for plotting
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(initial_conditions_tensor)
        data_loss = mse_loss(predictions, solutions_tensor)
        phy_loss = physics_loss(predictions)
        total_loss = data_loss + phy_loss
        total_loss.backward()
        optimizer.step()

        phy_loss_ihstory.append(phy_loss.item())
        data_loss_history.append(data_loss.item())
        loss_history.append(total_loss.item())
        epoch_history.append(epoch)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.1e}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch_history, y=phy_loss_ihstory, mode='lines', name='Physics Loss', line=dict(color='red', width=2)))
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
    test_initial_conditions = np.array([[x0, y0]])
    test_initial_conditions_tensor = torch.tensor(test_initial_conditions, dtype=torch.float32)

    # Predict solutions
    with torch.no_grad():
        test_predictions = model(test_initial_conditions_tensor).numpy()

    # Solve the ODE numerically for the same test initial conditions to get true solutions
    def van_der_pol(t, z):
        x, y = z
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        return [dxdt, dydt]

    sol = solve_ivp(van_der_pol, (0, t_end), test_initial_conditions[0], t_eval=t_eval, method='RK45')
    true_solutions = sol.y.T  # Shape: (time_steps, 2)

    MSE_x = (true_solutions[:, 0] - test_predictions[0, :, 0])**2
    MSE_y = (true_solutions[:, 1] - test_predictions[0, :, 1])**2

    # Plot results

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("x(t)", "y(t)"))
    fig.add_trace(go.Scatter(x=t_eval, y=true_solutions[:, 0], mode='lines', name="True Solution"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=test_predictions[0, :, 0], mode='lines', name="TBNN Prediction"), row=1, col=1)
    fig.update_layout(yaxis_title="x(t)", xaxis1=dict(showticklabels=False, showgrid=False)) 

    fig.add_trace(go.Scatter(x=t_eval, y=true_solutions[:, 1], mode='lines', name="True Solution"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=test_predictions[0, :, 1], mode='lines', name="TBNN Prediction"), row=2, col=1)
    fig.update_layout(xaxis2_title="Time t", yaxis2_title="y(t)")
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_solutions[:, 0], y=true_solutions[:, 1], mode='lines', name="True Solution"))
    fig.add_trace(go.Scatter(x=test_predictions[0, :, 0], y=test_predictions[0, :, 1], mode='lines', name="TBNN Prediction"))
    fig.update_layout(xaxis_title="x(t)", yaxis_title="y(t)", xaxis=dict(scaleanchor="y", scaleratio=1))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=MSE_x, name="x(t) MSE", opacity = 0.5))
    fig.add_trace(go.Histogram(x=MSE_y, name="y(t) MSE", opacity = 0.5))
    fig.update_layout(barmode='overlay', xaxis_title="MSE", yaxis_title="Count")
    st.plotly_chart(fig)
else:
    st.write("Please train the model first.")