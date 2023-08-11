from data_preparation import load_data, clean_data, merge_data, segment_data
from recommendation_engine import calculate_response_rate, build_recommendation_matrix, recommend_offer_for_demographic

def main():
    """The main function to run the entire data processing and recommendation pipeline."""
    portfolio, profile, transcript = load_data()
    profile_cleaned, transcript_cleaned = clean_data(profile, transcript)
    merged_data = merge_data(transcript_cleaned, profile_cleaned, portfolio)
    segmented_data = segment_data(merged_data)
    response_rate = calculate_response_rate(segmented_data)
    recommendation_matrix = build_recommendation_matrix(response_rate)
    print(recommendation_matrix)

if __name__ == "__main__":
    main()
