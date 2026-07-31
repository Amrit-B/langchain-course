
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion. 
                  """

    summary_template =  """ 
                        given the information {information} about a person I want you to create:
                        1. A short summary
                        2. two interesting facts about them
                        """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )



if __name__ == "__main__":
    main()
