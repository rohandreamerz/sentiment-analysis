from flask import Flask, request, render_template
from textblob import TextBlob
import string
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    
    stop_words = stopwords.words('english')

    title = request.form['song_title']

    lyrics = request.form['lyric'].lower()
   
    cleaned_lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))
                                    
    processed_lyrics = ' '.join([word for word in cleaned_lyrics.split() if word not in stop_words])
               
    score = SentimentIntensityAnalyzer().polarity_scores(processed_lyrics)
                                        
    compounds = round((1 + score['compound'])/2, 3)
                                                                        
    if  score['neg']>score['pos']:
        texts="The lyrics have a Negative sentiment 😥"
                                    
    elif score['neg'] < score['pos']:
        texts="The lyrics have a Positive Sentiment 🙂"
                                    
    else: 
        texts="The lyrics have a Neutral Sentiment 😐"    
    
    #tokenize
    tokenized_lyrics = word_tokenize(processed_lyrics, "english")
    
    #Lemmatize
    lemma_words = []
    for word in tokenized_lyrics:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
        
        
    #find emotions
    emotion_list = []
    with open('emotion.csv', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", ' ').strip()
            word, emotion = clear_line.split(' ')
            if word in lemma_words:
                emotion_list.append(emotion)

    w = Counter(emotion_list)
    
    
    a = dict(w)
    b = list(a.keys()) 
    c = list(a.values())
    
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    colors=['red','green','orange','yellow','magenta','cyan','black']
    ax1.bar(a.keys(), a.values(),color=colors)
    fig.autofmt_xdate()
    #plt.savefig('graph.png')
    plt.title('Emotions in the song', fontdict=None, loc='center', pad=None)
    plt.show()
    
   
    
    Maxa=max(emotion_list, key = emotion_list.count)
    mam = "The dominant emotion in the song is '{}'".format(Maxa.upper())
    Mina=min(emotion_list, key = emotion_list.count)
    mim = "The weakest emotion in the song is '{}'".format(Mina.upper())
    
    blob = TextBlob(processed_lyrics)
    polarity = blob.sentiment.polarity
  
    return render_template('submit.html', song_name='{}'.format(title), compound="{}".format(round(compounds*100,2)), text='{}'.format(texts),table='The emotions in the song are:\n {}'.format(b),maximum='{}'.format(mam),minimum='{}'.format(mim),polarity='{}'.format(round(polarity*100,2)))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
