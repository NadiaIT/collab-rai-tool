import pipeline



if __name__ == "__main__":
    sys_info = "A system that provides movie recommendations to users based on their watching history and ratings data. The system can receive recommendation requests and needs to reply with a list of recommended movies."
    goal = "f1"
    pipeline.generate_scenarios(sys_info=sys_info, goal=goal)

