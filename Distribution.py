import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def analyze_directory(data_directory):
    """Analyze the distribution of files in the given directory."""
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"The directory {data_directory} does not exist.")
    data = {}
    for subdirectory in os.listdir(data_directory):
        subdirectory_path = os.path.join(data_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            file_count = 0
            for item in os.listdir(subdirectory_path):
                item_path = os.path.join(subdirectory_path, item)
                if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_count += 1
            if file_count > 0:
                data[subdirectory] = file_count
    return data

def plot_histogram(data):
    """Plot a histogram of the file distribution."""
    keys = list(data.keys())
    values = list(data.values())
    n_bars = len(keys)
    colors = plt.cm.viridis(np.linspace(0, 1, n_bars))
    plt.figure(figsize=(10, 6))
    plt.bar(data.keys(), data.values(), color=colors)
    plt.xlabel('Subdirectories')
    plt.ylabel('Number of Images')
    plt.title('Image Distribution in Subdirectories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pie_chart(data):
    """Plot a pie chart of the file distribution."""
    plt.figure(figsize=(8, 8))
    plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Image Distribution in Subdirectories')
    plt.tight_layout()
    plt.show()

def plot_distribution(data):
    """Plot the distribution of files in subdirectories."""
    plot_histogram(data)
    plot_pie_chart(data)

def main():
    """main function for Distribution.py"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: python Distribution.py <data_directory>")
        data_directory = sys.argv[1]
        data = analyze_directory(data_directory)
        #Print datas
        if data:
            print("File distribution in subdirectories:")
            for subdir, count in data.items():
                print(f"{subdir}: {count} images")
        else:
            print("No files found in any subdirectory.")
        if data:
            plot_distribution(data)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()