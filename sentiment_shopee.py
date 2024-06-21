import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
import regex
import re
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from underthesea import word_tokenize, pos_tag, sent_tokenize
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')


#Text Processing
##LOAD EMOJICON
with open('emojicon.txt', 'r', encoding="utf8") as file:  
    emoji_lst = file.read().split('\n')
    emoji_dict = {}
    for line in emoji_lst:
        key, value = line.split('\t')
        emoji_dict[key] = str(value)

#################
#LOAD TEENCODE
with open('teencode.txt', 'r', encoding="utf8") as file:  
    teen_lst = file.read().split('\n')
    teen_dict = {}
    for line in teen_lst:
        key, value = line.split('\t')
        teen_dict[key] = str(value)

###############
#LOAD TRANSLATE ENGLISH -> VNMESE
with open('english-vnmese.txt', 'r', encoding="utf8") as file:  
    english_lst = file.read().split('\n')
    english_dict = {}
    for line in english_lst:
        key, value = line.split('\t')
        english_dict[key] = str(value)

################
#LOAD wrong words
with open('wrong-word.txt', 'r', encoding="utf8") as file:  
    wrong_lst = file.read().split('\n')

#################
#LOAD STOPWORDS
with open('vietnamese-stopwords.txt', 'r', encoding="utf8") as file:  
    stopwords_lst = file.read().split('\n')
  

def process_text_str(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(re.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    if isinstance(text, float):
        text = str(text)
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document



# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)



def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()



import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

# Áp dụng hàm chuẩn hóa cho văn bản
# print(normalize_repeated_characters(example))



def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document


stop_words = [
    'nhà hàng', 'nhà_hàng', 'quán', 'quán_ăn', 'ăn', 'đồ_ăn', 'bữa', 'buổi', 'trưa', 'tối', 'hôm nay', 'ngày mai'
    'sáng', 'thực_đơn', 'món', 'món_ăn', 'bàn', 'đặt bàn', 'đặt_bàn', 'nhân_viên', 'phục_vụ', 'dịch_vụ',
    'khách_hàng', 'khách', 'đồ_uống', 'món', 'món_ăn' 'giá', 'hóa_đơn',
    'phải_chăng', 'không_gian', 'trang_trí', 'chỗ', 'vị_trí', 'khu_vực', 'chất_lượng', 'số_lượng',
    'đề_nghị', 'gợi_ý', 'trải_nghiệm', 'thử', 'thưởng_thức', 'đánh_giá', 'sao', 'là', 'luôn',  'có',
    'nữa', 'nói', 'thấy', 'quá', 'cũng', 'làm', 'còn', 'bánh','người', 'thêm', 'khác', 'bạn', 'lại', 'nhìn',
    'phần', 'gọi', 'bên', 'chỉ', 'lên', 'gà', 'bán', 'chắc', 'phải',  'lúc', 'đi', 'kiểu', 'cơm', 'đặt', 'về',
    'chưa', 'kêu',  'mì', 'bánh_mì', 'hôm', 'thịt', 'tô', 'đến', 'hàng', 'nước chấm', 'tính', 'nước dùng', 'ngồi',
    'nước lèo', 'lần đầu', 'lấy', 'hộp', 'kèm', 'cứ', 'tiệm', 'nhà', 'tới', 'bò', 'sốt', 'đồ', 'bỏ', 'cơm gà', 'chủ',
    'cũng rất', 'ngày', 'thường', 'còn có', 'giống', 'chả', 'sáng', 'ghé', 'thấy cũng', 'thứ', 'chọn', 'toàn', 'có thêm',
    'vào', 'riêng', 'đem', 'giá cũng', 'hỏi', 'sẽ', 'loại', 'vô', 'ăn_ở', 'cách', 'phô', 'thức_ăn', 'chạy', 'giữ', 'cháo',
    'phở', 'đang', 'bún', 'tôm', 'thấy có', 'ốc', 'thịt bò', 'dĩa', 'cho', 'gọi phần', 'cực', 'còn lại','lúc_nào cũng',
    'đi người', 'uống', 'quận', 'xôi', 'chè', 'vẫn còn', 'như_vậy', 'mở', 'bảo', 'cùng', 'đưa', 'vịt', 'rất là', 'nước_mắm',
    'là thấy', 'đường', 'phần cơm', 'gửi', 'tầm', 'mặt', 'trước', 'được','lắm', 'rất', 'giá', 'khá', 'hơn', 'vẫn', 'hết', 'lần', 'mới', 'không_có',
    'có_thể', 'giờ', 'đều', 'biết', 'đúng', 'không_biết', 'không_phải', 'nướng', 'hơi', 'nhiều lần',
   'lần đầu', 'nghĩ', 'chiên', 'đủ', 'nhánh', 'ngoài', 'cá', 'điểm', 'nhưng_mà', 'hình', 'dịp', 'nơi', 'chiều','trên', 'trộn', 'cảm_giác',
   'liền', 'hình_như', 'miếng', 'nấu', 'từng', 'nằm', 'sẵn', 'số', 'mất', 'nhớ', 'chén', 'khoảng', 'lần lần', 'không_thấy',
   'đổi', 'cần', 'mẹ', 'ổ', 'nhận', 'gồm', 'lần đầu_tiên', 'khỏi', 'tí', 'không', 'nhiên', 'mặc_dù', 'giò', 'á', 'đầu',
   'nhận được', 'trời', 'giảm', 'việc', 'cực_kì', 'tiếp', 'đợi', 'rán', 'lúc_nào',  'có_điều', 'lầu', 'sợi', 'chẳng', 'cuốn', 'thành', 'xuống',
   'review', 'hồi', 'bịch', 'miệng', 'dùng', 'đùi', 'tây', 'không_bị', 'tên', 'cảm_nhận', 'nhóm',
   'trả', 'gỏi', 'hơn nhiều', 'nên', 'mới được']




def remove_stopword(text):
    ###### REMOVE stop words
    document = ' '.join('' if word in stop_words else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document


def clean_text_df(text):
  clean_text = text.apply(lambda x: process_text(x, emoji_dict, teen_dict, wrong_lst))
  clean_text = clean_text.apply(convert_unicode)
  clean_text = clean_text.apply(process_special_word)
  clean_text = clean_text.apply(normalize_repeated_characters)
  clean_text = clean_text.apply(process_postag_thesea)
  clean_text = clean_text.apply(remove_stopword)
  return clean_text

def clean_text_str(text):
  clean_text = process_text_str(text, emoji_dict, teen_dict, wrong_lst)
  clean_text = convert_unicode(clean_text)
  clean_text = process_special_word(clean_text)
  clean_text = normalize_repeated_characters(clean_text)
  clean_text = process_postag_thesea(clean_text)
  clean_text = remove_stopword(clean_text)
  return clean_text

def predict_sentiment(text):
    return '😊' if text == 1 else '😞'  

# Upload file
data = pd.read_csv('data_sentiment.csv')
data['Date'] = data['Time'].map(lambda x: x.split()[0])
data['Hour'] = data['Time'].map(lambda x: x.split()[1].split(':')[0])
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Hour'] = data['Hour'].astype(int)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

#load model
with open('pipeline.pkl', 'rb') as file:  
    sent_model = pickle.load(file)


#GUI
st.set_page_config(page_title='Sentiment Analysis', page_icon='📊', layout="centered")

menu = ["Giới thiệu", "Phân tích đánh giá", "Thông tin nhà hàng", "Xem thêm"]
choice = option_menu (
                    menu_title=None,
                    options = menu,
                    menu_icon = 'cast',
                    default_index = 0,
                    orientation = 'horizontal')

if choice == 'Giới thiệu': 
    st.title("Project 4")
    st.subheader("Nguyễn Thị Ngọc Trâm - Đào Minh Trí")
    st.subheader("🙂😐☹️ Sentiment Analysis")
    st.write("""
    - Sentiment Analysis là quá trình phân tích, đánh giá quan điểm của một người về một đối tượng nào đó (quan điểm mang tính tích cực, tiêu cực, hay trung tính,..). Quá trình này có thể thực hiện bằng việc sử dụng các tập luật (rule-based), sử dụng  Machine Learning hoặc phương pháp Hybrid (kết hợp hai  phương pháp trên).
    - Sentiment Analysis được ứng dụng nhiều trong thực tế, đặc biệt là trong hoạt động quảng bá kinh doanh. Việc phân tích đánh giá của người dùng về một sản phẩm xem họ đánh giá tiêu cực, tích cực hoặc đánh giá các hạn chế của sản phẩm sẽ giúp công ty nâng cao chất lượng sản phẩm và tăng cường hình ảnh của công ty, củngcố sự hài lòng của khách hàng.""")  
    st.image('sentimentanalysishotelgeneric-2048x803-1.jpg')
    st.subheader("🍽️ Sentiment Analysis trong ẩm thực")
    st.write("""
    - Để lựa chọn một nhà hàng/quán ăn mới chúng ta có xu hướng xem xét những bình luận từ những người đã thưởng thức để đưa ra quyết định có nên thử hay không?
    - Xây dựng hệ thống hỗ trợ nhà hàng/quán ăn phân loại các phản hồi của khách hàng  thành các nhóm: tích cực, tiêu cực, trung tính  dựa trên dữ liệu dạng văn bản.
    - Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/ sản phẩm.""")
    st.image('vn-11134513-7r98o-lugftthr8is27b.png')
    st.subheader("👩‍💻 Các bước thực hiện")
    st.image('project-10.jpg')

elif choice == 'Xem thêm':
    data_review = pd.read_csv('data_review_merge.csv')
    restaurant = pd.read_csv('1_Restaurants.csv')
    review = pd.read_csv('2_Reviews.csv')
    st.title("🔎 Data Review 📝")
    st.write("""
    - Dữ liệu được cung cấp sẵn trong tập tin 2_Reviews.csv với gần 30.000 mẫu gồm các thông tin:
    - ID (mã), User (người dùng), Time (thời gian đánh giá), Rating (điểm đánh giá), Comment (nội dung đánh giá), và IDRestaurant (mã nhà hàng)
    - Tập tin chứa thông tin về nhà hàng: 1_Restaurants.csv với hơn 1.600 mẫu gồm các thông tin:
    - 'ID (mã), Restaurant (tên nhà hàng), Address (địa chỉ), Time (giờ mở cửa), Price (khoảng giá), District(quận)""")
    st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="margin-bottom: 0;">1_Restaurants</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
    st.dataframe(restaurant)
    st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="margin-bottom: 0;">2_Reviews</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
    st.dataframe(review)
    st.markdown(f"**Tổng số lượng nhà hàng: {len(list(restaurant['Restaurant'].value_counts().index))}**")

    dis_res = restaurant.groupby('District')['Restaurant'].count().sort_values(ascending=False)
    top_10 = data_review.groupby('Restaurant')['Rating'].count().sort_values(ascending=False).head(10)
  
    st.markdown("**Tổng số lượng nhà hàng theo quận**")
    plt.figure(figsize=(10, 8))
    ax = dis_res.plot(kind='barh', x='District', y='Number of Restaurants', legend=False)
    ax.set_xlabel("Number of Restaurants")
    plt.title('Number of Restaurants by District')
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    st.pyplot(plt)    
   
    st.markdown("**Top 10 nhà hàng có số lượng đánh giá nhiều nhất**")
    plt.figure(figsize=(10, 8))
    axtop10 = top_10.plot(kind='barh')
    axtop10.set_xlabel("Number of Reviews")
    plt.title('Top 10 Restaurants with Most Reviews')
    for container in axtop10.containers:
        axtop10.bar_label(container, label_type='edge')
    st.pyplot(plt)    
 
    df_plot_sent = data['Rating_Score'].value_counts()
    st.markdown("**Phân phối Rating**")
    fig, ax = plt.subplots()
    ax.pie(df_plot_sent, labels=df_plot_sent.index, autopct='%1.1f%%', startangle=90,colors=['#66b3ff', '#ff9999', '#99ff99'])
    ax.axis('equal') 
    st.pyplot(fig)

    st.title("📈 Huấn luyện mô hình")
    st.subheader("Các bước thực hiện")
    st.image('project-10.jpg')
    st.write("""
    - Mô hình được huấn luyện với cả 3 tập train (OverSampling, UnderSampling và Tập train gốc)
    - Kết quả huấn luyện mô hình với dữ liệu gốc không tốt do dữ liệu bị mất cân bằng khá nặng
    - Qua 2 phương pháp cải thiện kết quả với việc chia tập train qua phương pháp OverSampling và UnderSampling
    - Phương pháp UnderSampling tập train với các mô hình đem lại kết quả khá tốt
    - Sau đây là kết quả của các mô hình khi được huấn luyện trên tập train được UnderSampling""")

    left_col6, right_col6 = st.columns(2)
    with left_col6:
        st.subheader('Logistic Regression')
        st.image('LR.JPG')
    with right_col6:
        st.subheader('NaiveBayes')
        st.image('NB.JPG')

    left_col7, right_col7 = st.columns(2)
    with left_col7:
        st.subheader('KNeighbors_3Classifier')
        st.image('KNN.JPG')
    with right_col7:
        st.subheader('KNeighbors_5Classifier')
        st.image('KNN5.JPG')

    left_col8, right_col8 = st.columns(2)
    with left_col8:
        st.subheader('DecisionTreeClassifier')
        st.image('DT.JPG')
    with right_col8:
        st.subheader('RandomForestClassifier')
        st.image('RF.JPG')

    left_col9, right_col9 = st.columns(2)
    with left_col9:
        st.subheader('SVC_Rbf')
        st.image('SVC_RBF.JPG')
    with right_col9:
        st.subheader('SVC_Linear')
        st.image('SVC_LINEAR.JPG')

    left_col10, right_col10 = st.columns(2)
    with left_col10:
        st.subheader('AdaBoostClassifier')
        st.image('ADA.JPG')
    with right_col10:
        st.subheader('XGBClassifier')
        st.image('XGB.JPG')


    st.write("**Nhận xét và lựa chọn mô hình**")
    st.write("""
    - Do dữ liệu bị mất cân bằng, Precision Score có thể sẽ thấp, ta có thể chỉ xem Recall Score và Accuracy Score để đánh giá
    - Mô hình SVC với kernel='rbf' với tập train được UnderSampling cho kết quả tốt nhất
    - Chỉ số Recall cho 2 nhãn đều khá cao (0.8 ~ 0.9), Accuracy Score đạt 0.88
    - Hơn nữa so sánh trực quan Cofusion Matrix cho thấy nhãn Positive và Negative dự đoán được tốt nhất trong tất cả các mô hình trên
    - Do đó ta sẽ chọn Mô hình SVC với kernel='rbf' RandomUnderSampling để dự đoán trên toàn bộ dữ liệu""")




elif choice == 'Phân tích đánh giá':
    st.subheader("🙂😐☹️ Phân tích đánh giá")
    st.write("""
    - Nhập 1 dòng đánh giá hoặc tải lên 1 bảng dữ liệu csv các đánh giá
    - Ắn nút phân tích, sẽ trả ra kết quả nhận xét đánh giá đó là Positive (Tích cực) hay Negative (Tiêu cực)""")
    st.markdown("##")
    st.subheader("Đánh giá")
    with st.form(key='TextForm'):
        text = st.text_area("Nhập 1 câu đánh giá vào đây")
        submit_button = st.form_submit_button(label = 'Phân tích')
    col1, col2 = st.columns(2)
    if submit_button:
        with col1:
            x_new = clean_text_str(text) 
            if isinstance(x_new, str):
                x_new = [x_new] 
            y_pred_new = sent_model.predict(x_new)       
            st.write(y_pred_new)
            if y_pred_new == 1:
                st.markdown("Positive 🙂")
            else:
                st.markdown("Negative ☹️")
        with col2:
            if y_pred_new == 1:
                st.image("smile.png")
            else:
                st.image("sad.png")

    st.subheader('Tải tệp')
    with st.form(key='dfform'):
        # Upload file
        uploaded_file = st.file_uploader("Tải tệp", type=['csv'])
        submit_button = st.form_submit_button(label = 'Phân tích')
        if uploaded_file is not None:
            st.markdown('---')
            df = pd.read_csv(uploaded_file, header=None,)
            st.markdown('Đánh giá của người dùng')
            st.dataframe(df)
            lines = df.iloc[:, 0]    
            if len(lines)>0:
                cleaned_lines = [clean_text_str(str(line)) for line in lines]             
                y_pred_new = sent_model.predict(cleaned_lines)
                df['Sentiment'] = y_pred_new
                df['Trạng thái'] = [predict_sentiment(text) for text in y_pred_new]
                st.markdown('Trạng thái đánh giá')
                st.dataframe(df)       


elif  choice == 'Thông tin nhà hàng':
    data_res = pd.read_csv('df_restaurants_fn.csv')
    res = st.multiselect(
                        "Lựa chọn nhà hàng:", 
                        options = data_res["Restaurant"].unique(),
                        max_selections = 1)
    if res:
        df_selection = data_res.query("Restaurant == @res")
        df_selection2 = data.query("Restaurant == @res")

        if df_selection.empty:
            st.warning("Chọn 1 nhà hàng từ hộp chọn trên")
        else:
            st.title(":bar_chart: Thông tin nhà hàng")
            st.markdown("##")

            name = df_selection["Restaurant"].values[0]
            rating_score = df_selection["Sentiment"].values[0]
            star_rating = round(df_selection["Rating"].values[0], 1)
            rating = '⭐️' * int(round(df_selection["Rating"].values[0], 0))
            price = df_selection["Price"].values[0]
            time = df_selection["Open_Time"].values[0]
            pos = df_selection["Positive"].values[0]
            neg = df_selection["Negative"].values[0]
            neu = df_selection["Neutral"].values[0]
            dis = df_selection["District"].values[0]
            add = df_selection["Address"].values[0]
            max_hour = df_selection["Most_Reviewed_Hour"].values[0]
            min_hour = df_selection["Min_Reviewed_Hour"].values[0]
            total_rat = int(pos + neg + neu)

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h1 style="margin-bottom: 0;">{name}</h1>
                </div>
                """, 
                unsafe_allow_html=True
            )

            leftcol1, rightcol1 = st.columns(2)
            with leftcol1:  
                st.markdown(
                f"""
                <div style="text-align: center;">
                    <h1>{star_rating}</h1>
                    <h4>{rating}</h4>
                    <p><strong>{total_rat} đánh giá</strong></p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            with rightcol1:
                df_plot_rat = df_selection.groupby(['Restaurant']).sum()[['9-10', '7-8', '5-6', '3-4', '1-2']]
                for restaurant, row in df_plot_rat.iterrows():
                    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
                    plt.barh(row.index, row.values, color=['#F9C70C'])
                    plt.xticks(np.arange(len(row)), [])  # Hide x-axis ticks
                    plt.gca().invert_yaxis()  # Invert y-axis to have the highest bar at the top
                    plt.box(False)
                    plt.gca().set_facecolor('none')
                    st.pyplot(plt)
            
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3>📍 Địa chỉ</h3>
                    <h3>{add}</h3>
                </div>
                """, 
                unsafe_allow_html=True)

            leftcol2, rightcol2 = st.columns([1.2, 1])
            with leftcol2:  
                st.markdown(
                    f"""
                        <h4>🕒Giờ hoạt động: {time}</h4>
                    """, 
                    unsafe_allow_html=True
                )
            with rightcol2:  
                st.markdown(
                    f"""
                        <h4>🏷️Giá: {price}</h4>
                    """,
                    unsafe_allow_html=True
                )
                
            leftcol3, rightcol3 = st.columns([1.2, 1])
            with leftcol2: 
                st.markdown(
                    f"""
                        <h4>🔼Đánh giá nhiều nhất: {max_hour} giờ</h4>
                    """, 
                    unsafe_allow_html=True
                )
            with rightcol2:
                st.markdown(
                    f"""
                        <h4>🔽Đánh giá ít nhất: {min_hour} giờ</h4>
                    """, 
                    unsafe_allow_html=True
                )
            st.markdown("""---""")

            left_col4, right_col4 = st.columns(2)
            with left_col4:
                st.subheader(f"☹️Negative: {neg} đánh giá")
                st.subheader(f"😊Positive: {pos} đánh giá")
                st.subheader(f"😐Neutral: {neu} đánh giá")
            with right_col4:
                df_plot_sent = df_selection.groupby(['Restaurant']).sum()[['Positive', 'Negative', 'Neutral']]
                for restaurant, row in df_plot_sent.iterrows():
                    plt.figure(figsize=(2, 2))
                    plt.pie(row, labels=row.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'none'}, colors=['#66b3ff', '#ff9999', '#99ff99'], textprops={'fontsize': 8})
                    plt.gca().set_facecolor('none')
                    st.pyplot(plt)
            st.markdown("""---""")

            left_col5, right_col5 = st.columns(2)
            with left_col5:
                st.subheader("Đánh giá Positive")
                pos_text = df_selection["comment_positive"].values[0]
                pw = WordCloud(width=400, height=200, background_color='white').generate(pos_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(pw, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            with right_col5:
                st.subheader("Đánh giá Negative")
                neg_text = df_selection["comment_negative"].values[0]
                nw = WordCloud(width=400, height=200, background_color='white').generate(neg_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(nw, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            left_col6, right_col6 = st.columns(2)
            with left_col6:
                df_pos_cm = df_selection2[(df_selection2['Sentiment'] == 'Positive')]
                df_pos_cm = df_pos_cm[['User', 'Time', 'Comment']]
                st.dataframe(df_pos_cm)
            with right_col6:
                df_neg_cm = df_selection2[(df_selection2['Sentiment'] == 'Negative')]
                df_neg_cm = df_neg_cm[['User', 'Time', 'Comment']]
                st.dataframe(df_neg_cm)


            # Plotting function
            def plot_sentiment_rating_trend(df_selection2, data):
                df_sub = data[data['Restaurant'] == df_selection2["Restaurant"].values[0]]
                grb = df_sub.groupby(['Year', 'Month']).agg({
                    'Sentiment': [('Positive', lambda x: (x == 'Positive').sum()),
                                ('Negative', lambda x: (x == 'Negative').sum())],
                    'Comment': 'count',
                    'Rating': 'mean'
                }).reset_index()
                grb.columns = ['Year', 'Month', 'Positive', 'Negative', 'num_comment', 'Rating']
                grb = grb.sort_values(by=['Year', 'Month'])
                grb['DateTime'] = pd.to_datetime(grb['Month'].astype(str) + '/' + grb['Year'].astype(str), format='%m/%Y')

                plt.figure(figsize=(10, 6))
                plt.plot(grb['DateTime'], grb['Rating'], marker='o', markersize=3, color='green')
                plt.title('Biểu đồ Rating trung bình theo thời gian')
                plt.xlabel('Thời gian')
                plt.ylabel('Rating')
                plt.ylim(0, 11)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

                plt.figure(figsize=(10, 6))
                plt.plot(grb['DateTime'], grb['Positive'], label='Positive', color='green', marker='o', markersize=3)
                plt.plot(grb['DateTime'], grb['Negative'], label='Negative', color='red', marker='o', markersize=3)
                plt.title('Biểu đồ các loại Sentiment theo thời gian')
                plt.xlabel('Thời gian')
                plt.ylabel('Số lượng')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

            plot_sentiment_rating_trend(df_selection2, data)
    else:
        st.warning("Chọn 1 nhà hàng từ hộp chọn trên")


                




        


           
