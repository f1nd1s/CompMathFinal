import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton, root_scalar
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox


class ComputationalMathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Computational Mathematics Interface")
        self.root.geometry("600x500")
        self.root.configure(bg='#2e2e2e')

        # Configure styles for UI elements
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background='#2e2e2e', foreground='white', font=('Arial', 12, 'bold'))
        style.configure("TButton", background='#4CAF50', foreground='white', font=('Arial', 12, 'bold'))
        style.map("TButton",
                  background=[('active', '#45a049')],
                  foreground=[('active', 'white')])
        style.configure("TCombobox", fieldbackground='#4CAF50', background='#4CAF50', foreground='white')

        # Dropdown menu for selecting computation methods
        self.method_label = ttk.Label(root, text="Select Method:")
        self.method_label.pack(pady=10)

        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(root, textvariable=self.method_var, width=50)
        self.method_dropdown['values'] = (
            'Task 1: Graphical Method', 'Task 2: Root-Finding Methods', 'Task 3: Jacobi Method',
            'Task 4: Matrix Inversion',
            'Task 5: Linear Curve Fitting', 'Task 6: Newton’s Interpolation', 'Task 7: Newton’s Derivative',
            'Task 8: Trapezoidal Rule')
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
        method = self.method_var.get()
        if method == 'Task 1: Graphical Method':
            self.coefficients = simpledialog.askstring("Input",
                                                       "Enter coefficients for f(x) (e.g., 1,-4,0,1 for x^3-4x+1):")
        elif method == 'Task 2: Root-Finding Methods':
            self.interval = simpledialog.askstring("Input", "Enter interval for root finding (e.g., 2,3):")
            self.coefficients = simpledialog.askstring("Input", "Enter coefficients for f(x) (e.g., 1,-4,0,1):")
        elif method == 'Task 3: Jacobi Method':
            self.coefficients = simpledialog.askstring("Input", "Enter coefficients matrix (e.g., 1,1,1;2,0,5;2,3,1):")
            self.constants = simpledialog.askstring("Input", "Enter constants vector (e.g., 6,-4,27):")
        elif method == 'Task 4: Matrix Inversion':
            self.matrix = simpledialog.askstring("Input", "Enter matrix (e.g., 4,-2,1;-2,4,-2;1,-2,4):")
        elif method == 'Task 5: Linear Curve Fitting':
            self.data_points = simpledialog.askstring("Input", "Enter data points (e.g., 1,2;2,3;3,5;4.7,11;5,9):")
        elif method == 'Task 6: Newton’s Interpolation':
            self.data_points = simpledialog.askstring("Input", "Enter data points x and y (e.g., 0,1,2,3;1,4,9,16):")
            self.value = simpledialog.askfloat("Input", "Enter value to interpolate (e.g., 1.5):")
        elif method == 'Task 7: Newton’s Derivative':
            self.data_points = simpledialog.askstring("Input", "Enter data points x and y (e.g., 0,1,2;1,8,27):")
            self.derivative_point = simpledialog.askfloat("Input", "Enter point to calculate derivative (e.g., 1):")
        elif method == 'Task 8: Trapezoidal Rule':
            self.integral_function = simpledialog.askstring("Input", "Enter function to integrate (e.g., x**2 + x):")
            self.interval = simpledialog.askstring("Input", "Enter interval for integration (e.g., 0,1):")
            self.subintervals = simpledialog.askinteger("Input", "Enter number of subintervals (e.g., 4):")

    def execute_method(self):
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
        elif method == 'Task 6: Newton’s Interpolation':
            self.task6()
        elif method == 'Task 7: Newton’s Derivative':
            self.task7()
        elif method == 'Task 8: Trapezoidal Rule':
            self.task8()
        else:
            self.result_text.insert(tk.END, "Please select a valid method.\n")

    def task1(self):
        try:
            coeffs = list(map(float, self.coefficients.split(',')))
            x = np.linspace(0, 3, 400)
            y = np.polyval(coeffs, x)

            root_approx = newton(lambda x: np.polyval(coeffs, x), 1)
            self.result_text.insert(tk.END, f'Approximate Root: {root_approx}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Graphical Method: {e}\n')

    def task2(self):
        try:
            a, b = map(float, self.interval.split(','))
            coeffs = list(map(float, self.coefficients.split(',')))
            f = lambda x: np.polyval(coeffs, x)
            root = bisect(f, a, b)
            self.result_text.insert(tk.END, f'Root found using Bisection Method: {root}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Root-Finding Method: {e}\n')


    def task3(self):
        try:
            A = np.array([list(map(float, row.split(','))) for row in self.coefficients.split(';')])
            b = np.array(list(map(float, self.constants.split(','))))
            x = np.zeros_like(b)

            def jacobi(A, b, x, tol=1e-6, max_iterations=100):
                D = np.diag(A)
                R = A - np.diagflat(D)
                for i in range(max_iterations):
                    x_new = (b - np.dot(R, x)) / D
                    if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                        return x_new
                    x = x_new
                return x

            solution = jacobi(A, b, x)
            self.result_text.insert(tk.END, f'Jacobi Method Solution: {solution}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Jacobi Method: {e}\n')

    def task4(self):
        try:
            A = np.array([list(map(float, row.split(','))) for row in self.matrix.split(';')])
            A_inv = np.linalg.inv(A)
            self.result_text.insert(tk.END, f'Inverse of Matrix:\n{A_inv}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Matrix Inversion: {e}\n')

    def task5(self):
        try:
            data_points = [list(map(float, point.split(','))) for point in self.data_points.split(';')]
            x, y = zip(*data_points)
            coeffs = np.polyfit(x, y, 1)
            self.result_text.insert(tk.END, f'Linear Fit Equation: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Linear Curve Fitting: {e}\n')

    def task6(self):
        try:
            x, y = [list(map(float, p.split(','))) for p in self.data_points.split(';')]
            polynomial = np.polynomial.Polynomial.fit(x, y, len(x) - 1)
            interpolated_value = polynomial(self.value)
            self.result_text.insert(tk.END, f'Interpolated value at x={self.value}: {interpolated_value}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Newton’s Interpolation: {e}\n')

    def task7(self):
        try:
            x, y = [list(map(float, p.split(','))) for p in self.data_points.split(';')]
            derivative = np.gradient(y, x)
            idx = np.searchsorted(x, self.derivative_point)
            self.result_text.insert(tk.END, f'Estimated Derivative at x={self.derivative_point}: {derivative[idx]}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Newton’s Derivative: {e}\n')

    def task8(self):
        try:
            f = lambda x: eval(self.integral_function, {'x': x, 'np': np})
            a, b = map(float, self.interval.split(','))
            x = np.linspace(a, b, self.subintervals + 1)
            y = f(x)
            integral = np.trapz(y, x)
            self.result_text.insert(tk.END, f'Integral approximation using Trapezoidal Rule: {integral}\n')
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Trapezoidal Rule: {e}\n')

    def plot_graph(self):
        method = self.method_var.get()
        if method == 'Task 1: Graphical Method':
            self.plot_task1()
        elif method == 'Task 5: Linear Curve Fitting':
            self.plot_task5()

    def plot_task1(self):
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
            plt.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            plt.grid(True, color='grey')
            plt.gca().set_facecolor('#1e1e1e')
            plt.gca().tick_params(colors='white')
            plt.show()
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Plotting Task 1: {e}\n')

    def plot_task5(self):
        try:
            data_points = [list(map(float, point.split(','))) for point in self.data_points.split(';')]
            x, y = zip(*data_points)
            coeffs = np.polyfit(x, y, 1)
            linear_fit = np.poly1d(coeffs)

            plt.figure(facecolor='#2e2e2e')
            plt.scatter(x, y, label='Data Points', color='#ff5722')
            plt.plot(x, linear_fit(x), label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}', color='#4CAF50')
            plt.title('Linear Curve Fitting', color='white')
            plt.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
            plt.grid(True, color='grey')
            plt.gca().set_facecolor('#1e1e1e')
            plt.gca().tick_params(colors='white')
            plt.show()
        except Exception as e:
            self.result_text.insert(tk.END, f'Error in Plotting Task 5: {e}\n')


if __name__ == "__main__":
    root = tk.Tk()
    app = ComputationalMathApp(root)
    root.mainloop()