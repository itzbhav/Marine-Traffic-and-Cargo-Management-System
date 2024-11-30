import sys
import pandas as pd
import numpy as np
import xarray as xr
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QFileDialog,QPushButton, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout,QMessageBox, QInputDialog, QVBoxLayout, QWidget, QTextEdit, QTextBrowser
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Load datasets
ports_data = pd.read_csv("D:/DOCUMENTS/SEM IV/ADSA PROJ/ports_data.csv", encoding="Latin-1")
ice_data = xr.open_dataset("D:/DOCUMENTS/SEM IV/ADSA PROJ/FMI-BAL-SEAICE_CONC-L4-NRT-OBS_1715739350131.nc")

def read_shipping_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    shipping_data = df.to_dict(orient='records')
    return shipping_data

def merge_sort(items, key):
    if len(items) <= 1:
        return items

    mid = len(items) // 2
    left_half = merge_sort(items[:mid], key)
    right_half = merge_sort(items[mid:], key)

    return merge(left_half, right_half, key)

def merge(left, right, key):
    sorted_list = []
    while left and right:
        if key(left[0]) > key(right[0]):
            sorted_list.append(left.pop(0))
        else:
            sorted_list.append(right.pop(0))

    sorted_list.extend(left)
    sorted_list.extend(right)
    return sorted_list

def knapsack_greedy(items, max_weight):
    sorted_items = merge_sort(items, key=lambda x: x["price ($)"] / x["weight (kg)"])

    total_value = 0
    total_weight = 0
    selected_items = []
    selected_pw_ratios = []

    for item in sorted_items:
        if total_weight + item["weight (kg)"] <= max_weight:
            total_weight += item["weight (kg)"]
            total_value += item["price ($)"]
            selected_items.append(item["name"])
            selected_pw_ratios.append(item["price ($)"] / item["weight (kg)"])

    return sorted_items, selected_items, selected_pw_ratios, total_value


# Preprocessing
def calculate_ice_cost(start_port, end_port, ice_data):
    start_lat = ports_data.loc[ports_data['name'] == start_port, 'lat'].iloc[0]
    start_lon = ports_data.loc[ports_data['name'] == start_port, 'lon'].iloc[0]
    end_lat = ports_data.loc[ports_data['name'] == end_port, 'lat'].iloc[0]
    end_lon = ports_data.loc[ports_data['name'] == end_port, 'lon'].iloc[0]

    start_lat_idx = np.abs(ice_data['latitude'] - start_lat).argmin().item()
    start_lon_idx = np.abs(ice_data['longitude'] - start_lon).argmin().item()
    end_lat_idx = np.abs(ice_data['latitude'] - end_lat).argmin().item()
    end_lon_idx = np.abs(ice_data['longitude'] - end_lon).argmin().item()

    ice_concentration_values = ice_data['ice_concentration'].values[0, start_lat_idx:end_lat_idx+1, start_lon_idx:end_lon_idx+1]
    avg_ice_concentration = np.nanmean(ice_concentration_values)

    if np.isnan(avg_ice_concentration):
        ice_cost = 0
    else:
        if avg_ice_concentration <= 30:
            ice_cost = 0
        elif 30 < avg_ice_concentration <= 60:
            ice_cost = 1
        else:
            ice_cost = 5

    return ice_cost if ice_cost is not None else 0

# Create Distance Matrix
ports = ports_data['name'].tolist()
n_ports = len(ports)
dist_matrix = np.full((n_ports, n_ports), np.inf)
np.fill_diagonal(dist_matrix, 0)

for i, start_port in enumerate(ports):
    for j, end_port in enumerate(ports):
        if start_port != end_port:
            start_lat = ports_data.loc[ports_data['name'] == start_port, 'lat'].iloc[0]
            start_lon = ports_data.loc[ports_data['name'] == start_port, 'lon'].iloc[0]
            end_lat = ports_data.loc[ports_data['name'] == end_port, 'lat'].iloc[0]
            end_lon = ports_data.loc[ports_data['name'] == end_port, 'lon'].iloc[0]
            distance = np.sqrt((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2)
            ice_cost = calculate_ice_cost(start_port, end_port, ice_data)
            total_cost = distance + ice_cost
            dist_matrix[i, j] = total_cost

# Implement Floyd-Warshall Algorithm
def floyd_warshall(dist_matrix):
    n = dist_matrix.shape[0]
    next_hop = np.zeros((n, n), dtype=int) - 1  # To store the next node in the path

    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i, j] != np.inf:
                next_hop[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
                    next_hop[i, j] = next_hop[i, k]

    return dist_matrix, next_hop

dist_matrix, next_hop = floyd_warshall(dist_matrix)

def reconstruct_path(next_hop, i, j):
    if next_hop[i, j] == -1:
        return []
    path = [i]
    while i != j:
        i = next_hop[i, j]
        path.append(i)
    return path

# Identify ports with intermediate stops
intermediate_ports = []
for i, start_port in enumerate(ports):
    for j, end_port in enumerate(ports):
        if i != j:
            path = reconstruct_path(next_hop, i, j)
            if len(path) > 2:  # Means there is at least one intermediate port
                intermediate_ports.append((start_port, end_port, [ports[k] for k in path[1:-1]]))

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, subplot_kw={'projection': ccrs.PlateCarree()})
        super().__init__(fig)
        self.setParent(parent)

    def plot_shortest_path_on_map(self, ports_data, path_ports):
        self.ax.clear()
        self.ax.coastlines(resolution='10m')
        self.ax.add_feature(cfeature.BORDERS)
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.LAKES)
        self.ax.add_feature(cfeature.RIVERS)

        lats = []
        lons = []

        for port in path_ports:
            lat = ports_data.loc[ports_data['name'] == port, 'lat'].iloc[0]
            lon = ports_data.loc[ports_data['name'] == port, 'lon'].iloc[0]
            lats.append(lat)
            lons.append(lon)

        # Plot the path
        self.ax.plot(lons, lats, color='red', linewidth=2, marker='o', transform=ccrs.PlateCarree())
        for i, port in enumerate(path_ports):
            self.ax.text(lons[i], lats[i], port, transform=ccrs.PlateCarree())

        self.ax.set_title('Shortest Path between Ports')
        self.draw()
class StockIntakeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cargo Allocation")
        self.setGeometry(100, 100, 500, 500)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        # Heading Label
        heading_label = QLabel("Cargo Allocation", self)
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-top: 20px; margin-bottom: 20px;")
        central_layout.addWidget(heading_label)

        # Input Fields
        input_layout = QVBoxLayout()
        central_layout.addLayout(input_layout)

        self.start_port_entry = QLineEdit(self)
        self.start_port_entry.setPlaceholderText("Enter the start port")
        input_layout.addWidget(self.start_port_entry)

        self.end_port_entry = QLineEdit(self)
        self.end_port_entry.setPlaceholderText("Enter the end port")
        input_layout.addWidget(self.end_port_entry)

        self.max_weight_entry = QLineEdit(self)
        self.max_weight_entry.setPlaceholderText("Enter the max weight limit of the ship")
        input_layout.addWidget(self.max_weight_entry)

        # Calculate Button
        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.setFont(QFont('Arial', 14))
        self.calculate_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.calculate_button.clicked.connect(self.display_results)
        input_layout.addWidget(self.calculate_button)

        # Result Display
        self.result_text = QTextBrowser(self)
        central_layout.addWidget(self.result_text)

    def display_results(self):
        start_port = self.start_port_entry.text()
        end_port = self.end_port_entry.text()
        max_weight_limit = int(self.max_weight_entry.text())

        file_path = "D:/DOCUMENTS/SEM IV/ADSA PROJ/shipping_data(1).xlsx"
        shipping_data = read_shipping_data(file_path)
        filtered_items = [item for item in shipping_data if str(item["end_port"]).strip() == end_port.strip()]
        sorted_items, selected_names, selected_pw_ratios, total_value = knapsack_greedy(filtered_items, max_weight_limit)

        self.result_text.clear()
        self.result_text.append("Available Items:")
        for item in sorted_items:
            self.result_text.append(f"Name: {item['name']}\tWeight: {item['weight (kg)']}\tPrice: {item['price ($)']}\tPW-Ratio: {item['price ($)'] / item['weight (kg)']}")

        self.result_text.append(f"\nMaximum Total Value: {total_value}")
        self.result_text.append(f"Maximum Weight Limit: {max_weight_limit}")

        if total_value == 0:
            self.result_text.append("Weight limit is zero or no items selected")
        else:
            self.result_text.append("Selected Items within Weight Limit:")
            for name, pw_ratio in zip(selected_names, selected_pw_ratios):
                self.result_text.append(f"Name: {name}\tPW-Ratio: {pw_ratio}")
class FindRouteWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Find Route')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.start_port_label = QLabel("Start Port:")
        layout.addWidget(self.start_port_label)
        self.start_port_input = QLineEdit()
        layout.addWidget(self.start_port_input)

        self.end_port_label = QLabel("End Port:")
        layout.addWidget(self.end_port_label)
        self.end_port_input = QLineEdit()
        layout.addWidget(self.end_port_input)

        self.calculate_button = QPushButton("Calculate Shortest Path")
        self.calculate_button.setFont(QFont('Arial', 14))
        self.calculate_button.setStyleSheet("background-color: #4CAF50; color: white;")
        layout.addWidget(self.calculate_button)

        self.result_text = QTextEdit()
        layout.addWidget(self.result_text)

        self.canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.calculate_button.clicked.connect(self.calculate_shortest_path)

    def calculate_shortest_path(self):
        start_port = self.start_port_input.text().strip().upper()
        end_port = self.end_port_input.text().strip().upper()

        start_index = ports.index(start_port)
        end_index = ports.index(end_port)

        shortest_path_indices = reconstruct_path(next_hop, start_index, end_index)
        shortest_path_ports = [ports[i] for i in shortest_path_indices]
        shortest_distance = dist_matrix[start_index, end_index]

        result_text = f"Shortest Path: {shortest_path_ports}\nShortest Distance: {shortest_distance}"

        # Update the result text widget
        self.result_text.setPlainText(result_text)

        # Update the plot
        self.canvas.plot_shortest_path_on_map(ports_data, shortest_path_ports)

class ViewDatasetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DATASET VIEWER")
        self.setGeometry(100, 100, 400, 300)
        self.label = QLabel("Select a file to read:", self)
        self.label.setGeometry(10, 10, 200, 30)

        self.button = QPushButton("Browse", self)
        self.button.setGeometry(10, 50, 100, 30)
        self.button.clicked.connect(self.browse_file)

        self.text = QTextEdit(self)
        self.text.setGeometry(10, 90, 780, 400)

        self.filepath = None

    def browse_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Open file", "", "CSV files (*.csv);;Excel files (*.xlsx);;All files (*.*)")
        if self.filepath:
            self.read_file()

    def read_file(self):
        try:
            if self.filepath.endswith('.csv'):
                df = pd.read_csv(self.filepath, low_memory=False)
            elif self.filepath.endswith('.xlsx'):
                df = pd.read_excel(self.filepath)
            else:
                QMessageBox.critical(self, "Unsupported file", "Only CSV and Excel files are supported.")
                return

            self.text.clear()
            self.text.insertPlainText(df.head(30).to_string())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while reading the file: {str(e)}")


class LoginPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Login Page')
        self.setGeometry(100, 100, 400, 300)
        self.setFixedSize(400, 300)

        # Background Image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap("C:/Users/bhava/Downloads/bg1.jpg"))
        self.background_label.setGeometry(0, 0, 400, 300)

        # Company Name Label
        self.company_label = QLabel('Company Name:', self)

        # Username Label
        self.username_label = QLabel('Username:', self)

        # Password Label
        self.password_label = QLabel('Password:', self)

        # Company Name Input
        self.company_input = QLineEdit(self)

        # Username Input
        self.username_input = QLineEdit(self)

        # Password Input
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        # Login Button
        self.login_button = QPushButton('Login', self)
        self.login_button.clicked.connect(self.on_login_button_clicked)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.company_label)
        layout.addWidget(self.company_input)
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        self.setLayout(layout)

    def on_login_button_clicked(self):
        # Check login credentials
        if self.company_input.text() == 'livefleet' and self.username_input.text() == 'admin' and self.password_input.text() == 'admin':
            # Navigate to the next page (replace with your next page implementation)
            self.next_page = NextPage()
            self.next_page.showFullScreen()
            self.hide()
        else:
            # Display error message or handle invalid login credentials
            print("Invalid login credentials")

class NextPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DASHBOARD')

        # Background Image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap("C:/Users/bhava/Downloads/bg2.jpg"))
        self.background_label.setScaledContents(True)  # Scale the image to fit the window
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.lower()  # Ensure background label is behind other widgets

        # Buttons
        self.button1 = QPushButton('View Dataset', self)
        self.button1.setStyleSheet("font-size: 15px;")
        self.button2 = QPushButton('Find Route', self)
        self.button2.setStyleSheet("font-size: 15px;")
        self.button3 = QPushButton('Stock Intake', self)
        self.button3.setStyleSheet("font-size: 15px;")

        # Set buttons size
        self.button1.setFixedSize(490, 350)
        self.button2.setFixedSize(490, 350)
        self.button3.setFixedSize(490, 350)

        # Set buttons icons
        self.button1.setIcon(QIcon("C:/Users/bhava/Downloads/datasetlogo.png"))  # Replace with the path to your logo
        self.button2.setIcon(QIcon("C:/Users/bhava/Downloads/route.png"))  # Replace with the path to your logo
        self.button3.setIcon(QIcon("C:/Users/bhava/Downloads/cargo.png"))  # Replace with the path to your logo

        # Set icon sizes
        self.button1.setIconSize(self.button1.size())
        self.button2.setIconSize(self.button2.size())
        self.button3.setIconSize(self.button3.size())

        button_style = """
            QPushButton {
                background-color: #d8bfd8;  # Light purple color
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #d3a4ff;  # Light purple hover effect
            }
        """
        self.button1.setStyleSheet(button_style)
        self.button2.setStyleSheet(button_style)
        self.button3.setStyleSheet(button_style)

        # Connect buttons to methods
        self.button1.clicked.connect(self.open_view_dataset_window)
        self.button2.clicked.connect(self.open_find_route_window)
        self.button3.clicked.connect(self.open_stock_intake_window)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button1)
        button_layout.addStretch(1)  # Add stretchable space before and after button2
        button_layout.addWidget(self.button2)
        button_layout.addStretch(1)
        button_layout.addWidget(self.button3)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addStretch(1)  # Add stretchable space at the top
        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)  # Add stretchable space at the bottom
        self.setLayout(main_layout)

    def resizeEvent(self, event):
        # Resize background to fit the window
        self.background_label.setGeometry(0, 0, self.width(), self.height())

    def open_view_dataset_window(self):
        # Open View Dataset window
        self.view_dataset_window = ViewDatasetWindow()
        self.view_dataset_window.show()

    def open_find_route_window(self):
        # Open Find Route window
        self.find_route_window = FindRouteWindow()
        self.find_route_window.show()

    def open_stock_intake_window(self):
        # Open Stock Intake window
        self.stock_intake_window = StockIntakeWindow()
        self.stock_intake_window.show()


        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_page = LoginPage()
    login_page.show()
    sys.exit(app.exec_())
       
