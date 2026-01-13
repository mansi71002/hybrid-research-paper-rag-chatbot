def expand_query(llm, query):
    prompt = f"""
    Write a detailed academic explanation for:
    {query}
    """
    return llm.invoke(prompt).content
