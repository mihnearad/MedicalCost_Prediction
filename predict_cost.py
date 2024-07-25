import joblib
import numpy as np

# Load the model
model = joblib.load('medical_cost_model.pkl')

def preprocess_input(input_features):
    """
    Preprocesses the input features to match the training preprocessing steps.
    Since we are not using a scaler, this function just reshapes the input.
    """
    input_array = np.array(input_features).reshape(1, -1)
    return input_array

def predict_cost(input_features):
    """
    Predicts the medical cost based on input features.
    Args:
    input_features (list): List of input features in the same order as the training data.
    
    Returns:
    float: Predicted medical cost.
    """
    input_array = preprocess_input(input_features)
    
    # Predict using the model
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == "__main__":
    print("Predicting Medical Costs")
    
    # Example input features
    # Age, Sex (1 for male, 0 for female), BMI, Children, Smoker (1 for yes, 0 for no), Region (numerical values)
    
    try:
        age = float(input("Please enter your age: "))
        gender = int(input("Please enter your gender (0 female, 1 male): "))
        bmi = float(input("Please enter your BMI: "))
        kids = int(input("Please enter the number of children you have: "))
        smoke = int(input("Do you smoke? (0 no, 1 yes): "))
        region = int(input("Please enter your region (0 for southwest, 1 for southeast, 2 for northwest, 3 for northeast): "))
        
        example = [age, gender, bmi, kids, smoke, region]
        # example_1 = [25, 1, 29.7, 0, 0, 1]  # 1: northwest
        # example_2 = [55, 1, 29.8, 3, 0, 0]  # 0: northeast
        
        print("Example 1")
        
        # Predict and print results
        predicted_cost = predict_cost(example)
        # predicted_cost_1 = predict_cost(example_1)
        # predicted_cost_2 = predict_cost(example_2)
        
        print(f'Predicted Medical Cost for Input Example: {predicted_cost}')
        # print(f'Predicted Medical Cost for Example 1: {predicted_cost_1}')
        # print(f'Predicted Medical Cost for Example 2: {predicted_cost_2}')
    
    except ValueError as e:
        print("Invalid input:", e)
