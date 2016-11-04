import Qnabot

bot = Qnabot.Qnabot(10, 10, 10)
# bot.grap_text(
print '\ngen_sentence:\n'
bot.gen_sentences('sentences.txt', 'raw_text.txt', 2)
bot.gen_sentences('sentences_Qs.txt', 'raw_text_Qs.txt', 2)
print '\nword2vec_train\n'
bot.word2vec_train('sentences.txt', min_count=2, workers=8)
bot.word2vec_train('sentences_Qs.txt', min_count=2, workers=8)
print '\nNN_train\n'
bot.NN_train('raw_text_Qs.txt', 'raw_text_As.txt',
              regul=0.00, verbose=True)
Qs = ['Is Python case senstive?', 'How to comment in python?']
print '\nNN_predict\n'
y, ans = bot.predict(Qs, 2)
print ans
