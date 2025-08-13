import os
from smolagents_crew import Agent, Task, Crew, TaskDependency
from smolagents import CodeAgent, TransformersModel, OpenAIServerModel, InferenceClientModel, MLXModel

# Create your AI dream team! 🤖
agent1 = Agent("translator", agent_instance=CodeAgent, model=MLXModel('md-nishat-008/TigerLLM-1B-it'), tools=[])
agent2 = Agent("coder", agent_instance=CodeAgent, model=MLXModel('mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit'), tools=[])

# Define their missions with smart dependencies 📋
task1 = Task(
    name="translator",
    agent=agent1,
    prompt_template="Translate the bengali text to english. Keep the `Example:` as <<Example>>. Do not provide python function. Do not write any code. \n<<Bengali Text>> {instruction}",
    result_key="translation_result"
)

task2 = Task(
    name="coder",
    agent=agent2,
    prompt_template="Write a python function with the function signature from <<Example>>.\n <<Instruction>> {translation_result}",
    dependencies=[TaskDependency("translator", "translation_result")]
)


# Assemble and launch your crew! 🚀
crew = Crew(
    agents={"translator": agent1, "coder": agent2},
    tasks=[task1, task2],
    initial_context={"instruction": '''একটি প্রদত্ত স্ট্রিং-এ প্রথম পুনরাবৃত্ত অক্ষর খুঁজে পেতে একটি পাইথন ফাংশন লিখুন।
Exammple:
first_repeated_char(s)'''}
)

results = crew.execute()