import argparse

from summarizers.LSASummarizer import LSASummarizer


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("text", type=str,
                        help="text to summarize")
    parser.add_argument("-l", "--language", type=str,
                        help="language of input text", default="english")
    parser.add_argument("-sl", "--summary-length", type=int, help="number of summary sentences", default=3)

    args = parser.parse_args()
    summarizer = LSASummarizer(args.language)
    summary = summarizer.summarize(args.text, args.summary_length)
    print("Original Text:")
    print(args.text)
    print("\n\nSummary:")
    print(summary)


if __name__ == '__main__':
    main()
