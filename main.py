import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend for Matplotlib to work with Tkinter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect, newton  # Import numerical root-finding methods
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox  # Import Tkinter components for GUI

class ComputationalMathApp:
    def __init__(self, root):
        """Initialize the GUI application for computational mathematics."""
        self.root = root
        self.root.title("Computational Mathematics Interface")
        self.root.geometry("600x500")
        self.root.configure(bg='#2e2e2e')  # Set background color for dark mode UI

        # Configure styles for UI elements (dark theme)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background='#2e2e2e', foreground='white', font=('Arial', 12, 'bold'))
        style.configure("TButton", background='#4CAF50', foreground='white', font=('Arial', 12, 'bold'))
        style.map("TButton", background=[('active', '#45a049')], foreground=[('active', 'white')])
        style.configure("TCombobox", fieldbackground='#4CAF50', background='#4CAF50', foreground='white')

        # Dropdown menu for selecting computation methods
        self.method_label = ttk.Label(root, text="Select Method:")
        self.method_label.pack(pady=10)

        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(root, textvariable=self.method_var, width=50)
        self.method_dropdown['values'] = (
            'Task 1: Graphical Method', 'Task 2: Root-Finding Methods', 'Task 3: Jacobi Method',
            'Task 4: Matrix Inversion', 'Task 5: Linear Curve Fitting', 'Task 6: Newton’s Interpolation',
            'Task 7: Newton’s Derivative', 'Task 8: Trapezoidal Rule'
        )
        self.method_dropdown.pack(pady=10)

        # Buttons for user interaction
        self.input_button = ttk.Button(root, text="Input Parameters", command=self.input_parameters)
        self.input_button.pack(pady=10)

        self.execute_button = ttk.Button(root, text="Execute Calculation", command=self.execute_method)
        self.execute_button.pack(pady=10)

        self.plot_button = ttk.Button(root, text="Plot Graph/Table", command=self.plot_graph)
        self.plot_button.pack(pady=10)

        # Text box for displaying results
        self.result_text = tk.Text(root, height=15, width=70, bg='#1e1e1e', fg='white', insertbackground='white',
                                   font=('Courier', 12))
        self.result_text.pack(pady=10)

    def input_parameters(self):
        """Handle input dialogs based on the selected computational method."""
        method = self.method_var.get()
        if method == 'Task 1: Graphical Method':
            self.coefficients = simpledialog.askstring("Input", "Enter coefficients for f(x) (e.g., 1,-4,0,1 for x^3-4x+1):")
        elif method == 'Task 2: Root-Finding Methods':
            self.interval = simpledialog.askstring("Input", "Enter interval for root finding (e.g., 2,3):")
            self.coefficients = simpledialog.askstring("Input", "Enter coefficients for f(x) (e.g., 1,-4,0,1):")
        elif method == 'Task 3: Jacobi Method':
            self.coefficients = simpledialog.askstring("Input", "Enter coefficient matrix (e.g., 1,1,1;2,0,5;2,3,1):")
            self.constants = simpledialog.askstring("Input", "Enter constants vector (e.g., 6,-4,27):")
        elif method == 'Task 4: Matrix Inversion':
            self.matrix = simpledialog.askstring("Input", "Enter matrix (e.g., 4,-2,1;-2,4,-2;1,-2,4):")
        elif method == 'Task 5: Linear Curve Fitting':
            self.data_points = simpledialog.askstring("Input", "Enter data points (e.g., 1,2;2,3;3,5;4.7,11;5,9):")

    def execute_method(self):
        """Call the appropriate method based on the user's selection."""
        method = self.method_var.get()
        self.result_text.delete(1.0, tk.END)  # Clear previous results

        if method == 'Task 1: Graphical Method':
            self.task1()
        elif method == 'Task 2: Root-Finding Methods':
            self.task2()
        elif method == 'Task 3: Jacobi Method':
            self.task3()
        elif method == 'Task 4: Matrix Inversion':
            self.task4()
        elif method == 'Task 5: Linear Curve Fitting':
            self.task5()
        else:
            self.result_text.insert(tk.END, "Please select a valid method.\n")

    def task1(self):
        """Graphical method for finding approximate roots of a polynomial."""
        try:
            coeffs = list(map(float, self.coefficients.split(',')))
            x = np.linspace(0, 3, 400)  # Generate x values
            y = np.polyval(coeffs, x)   # Compute f(x)

            root_approx = newton(lambda x: np.polyval(coeffs, x), 1)  # Newton's method for root approximation
            self.result_text.insert(tk.END, f'Approximate Root: {root_approx}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Graphical Method: {e}\n')

    def task2(self):
        """Root-finding using the Bisection Method."""
        try:
            a, b = map(float, self.interval.split(','))  # Get interval
            coeffs = list(map(float, self.coefficients.split(',')))  # Get coefficients
            f = lambda x: np.polyval(coeffs, x)  # Define function f(x)

            root = bisect(f, a, b)  # Apply Bisection Method
            self.result_text.insert(tk.END, f'Root found using Bisection Method: {root}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Root-Finding Method: {e}\n')

    def task3(self):
        """Solve a system of equations using the Jacobi method."""
        try:
            A = np.array([list(map(float, row.split(','))) for row in self.coefficients.split(';')])
            b = np.array(list(map(float, self.constants.split(','))))
            x = np.zeros_like(b)  # Initialize solution

            def jacobi(A, b, x, tol=1e-6, max_iterations=100):
                """Jacobi iterative method for solving Ax = b."""
                D = np.diag(A)  # Diagonal elements
                R = A - np.diagflat(D)  # Remaining elements
                for _ in range(max_iterations):
                    x_new = (b - np.dot(R, x)) / D  # Iterative update
                    if np.linalg.norm(x_new - x, ord=np.inf) < tol:  # Convergence check
                        return x_new
                    x = x_new
                return x

            solution = jacobi(A, b, x)
            self.result_text.insert(tk.END, f'Jacobi Method Solution: {solution}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Jacobi Method: {e}\n')

    def task4(self):
        matrix = np.array([list(map(float, row.split(','))) for row in self.matrix.split(';')])
        inverse = np.linalg.inv(matrix)
        self.result_text.insert(tk.END, f'Matrix Inversion Result:\n{inverse}\n')

    def task5(self):
        points = np.array([list(map(float, pair.split(','))) for pair in self.data_points.split(';')])
        x, y = points[:, 0], points[:, 1]
        coef = np.polyfit(x, y, 1)
        self.result_text.insert(tk.END, f'Linear Fit: y = {coef[0]:.2f}x + {coef[1]:.2f}\n')

    def task6(self):
        x = list(map(float, self.x_values.split(',')))
        y = list(map(float, self.y_values.split(',')))
        interpolation = interp1d(x, y, kind='linear')
        estimate = interpolation(1.5)
        self.result_text.insert(tk.END, f'Interpolated Value at x=1.5: {estimate}\n')

    def task7(self):
        x = np.array(list(map(float, self.x_values.split(','))))
        y = np.array(list(map(float, self.y_values.split(','))))
        derivative = (y[2] - y[1]) / (x[2] - x[1])
        self.result_text.insert(tk.END, f'Estimated Derivative at x=1: {derivative}\n')

    def task8(self):
        f = lambda x: x ** 2 + x
        a, b, n = 0, 1, int(self.sub_intervals)
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        integral = (h / 2) * (y[0] + 2 * sum(y[1:n]) + y[n])
        exact = (1 ** 3 / 3 + 1 ** 2 / 2) - (0 ** 3 / 3 + 0 ** 2 / 2)
        self.result_text.insert(tk.END, f'Trapezoidal Approximation: {integral}\nExact Integral: {exact}\n')

    def plot_graph(self):
        """Handle graph plotting for selected computational tasks."""
        method = self.method_var.get()
        if method == 'Task 1: Graphical Method':
            self.plot_task1()

    def plot_task1(self):
        """Plot polynomial function for graphical root-finding method."""
        try:
            coeffs = list(map(float, self.coefficients.split(',')))
            x = np.linspace(0, 3, 400)
            y = np.polyval(coeffs, x)

            plt.figure(facecolor='#2e2e2e')
            plt.plot(x, y, label='f(x)', color='#4CAF50')
            plt.axhline(0, color='grey', linestyle='--')
            plt.title('Graphical Method for Root Finding', color='white')
            plt.xlabel('x', color='white')
            plt.ylabel('f(x)', color='white')
            plt.legend()
            plt.show()
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Plotting Task 1: {e}\n')

if __name__ == "__main__":
    root = tk.Tk()
    app = ComputationalMathApp(root)
    root.mainloop()
