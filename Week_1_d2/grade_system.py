try:
    mark = float(input("Enter your mark (0-100): "))

    if mark < 0 or mark > 100:
        raise ValueError("Mark must be between 0 and 100.")

    if mark >= 90 and mark <= 100:
        grade = "A"
    elif mark >= 80:
        grade = "B"
    elif mark >= 70:
        grade = "C"
    elif mark >= 60:
        grade = "D"
    else:
        grade = "E"

    print(f"Entered mark: {mark}")
    print(f"Grade: {grade}")

except ValueError as e:
    print(f"Error: {e}")