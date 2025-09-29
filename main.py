import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

billing_agent = Agent(
    name="Billing Agent",
    instructions="You are a helpful agent to handle related to billing queries"
)

security_agent= Agent(
    name="Security Agent",
    instructions="You are a helpful agent to handle related to security queries"
)

agent = Agent(
        name="System Support Agent",
        instructions="""
            you are a helpful customer support agent for a software company.
            your task is to assist users with their software-related issues and provide solutions in a friendly and professional manner.
            always ask clarifying questions if the user's issue is not clear.
            if the user asks for information outside of software support, politely inform them that you can only assist with software-related issues.
            ensure that your responses are concise and to the point.
        """,
        model=model,
)

result = Runner.run(
    agent,
    input="My software keeps crashing when I try to open it. Can you help?",
    run_config=config
    )