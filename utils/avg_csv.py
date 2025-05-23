import csv
import argparse

def calculate_average_from_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        numbers = [float(row[0]) for row in reader if row]  
        
        if not numbers:  
            return 0
        
        average = sum(numbers) / len(numbers)
        return average

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=str)
    args = parser.parse_args()

    average = calculate_average_from_csv(args.eval_csv)
    print(average)