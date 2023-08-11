def calculate_response_rate(merged_data):
    """Calculate the response rate for each demographic and offer type.
    
    Args:
        merged_data (DataFrame): The merged dataset.
        
    Returns:
        DataFrame: A dataframe showing the response rate for each demographic and offer type.
    """
    response_rate = merged_data.groupby(['age_group', 'offer_type']).apply(
        lambda x: x[x['event'] == 'offer completed'].shape[0] / x.shape[0]).reset_index()
    response_rate.columns = ['age_group', 'offer_type', 'response_rate']
    return response_rate

def build_recommendation_matrix(response_rate):
    """Build the recommendation matrix based on response rates.
    
    Args:
        response_rate (DataFrame): The response rate for each demographic and offer type.
        
    Returns:
        DataFrame: The recommendation matrix.
    """
    recommendation_matrix = response_rate.pivot(index='age_group', columns='offer_type', values='response_rate')
    return recommendation_matrix

def recommend_offer_for_demographic(recommendation_matrix, age_group):
    """Recommend the best offer type for a given age group.
    
    Args:
        recommendation_matrix (DataFrame): The recommendation matrix.
        age_group (str): The age group for which to recommend an offer.
        
    Returns:
        str: The recommended offer type.
    """
    return recommendation_matrix.loc[age_group].idxmax()
