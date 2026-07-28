import streamlit as st

st.title("Power Calculator 2x , 3x , 5x")
st.write("Enter a number to calculate its square, cube, and fifth power.")

# User input
n = st.number_input("Enter an integer", value=1, step=1)

# Calculate results
square = n ** 2
cube = n ** 3
fifth_power = n ** 5

# Display results
st.subheader("Results")
st.write(f"Square: {square}")
st.write(f"Cube: {cube}")
st.write(f"Fifth Power: {fifth_power}")