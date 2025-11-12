from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# ==================== ML ИМПОРТЫ ====================

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    import numpy as np
    ML_AVAILABLE = True
    print("✅ ML модели загружены")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML модели недоступны: {e}")

# ==================== ГИБРИДНЫЙ АНАЛИЗАТОР ====================

def hybrid_sentiment_analyzer(text):
    """Гибридный анализатор: rule-based + ML"""
    text_lower = text.lower().strip()
    
    # Быстрая проверка пустого текста
    if not text_lower or len(text_lower) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # СИЛЬНЫЕ ИНДИКАТОРЫ (правила - ВЫСОКИЙ ПРИОРИТЕТ)
    strong_negative = [
        'хуйня', 'говно', 'пиздец', 'дерьмо', 'отстой', 'отвратительно', 'отвратительный',
        'ужасный', 'кошмар', 'провал', 'мудак', 'гандон', 'ублюдок', 'пидорас', 'залупа'
    ]
    
    strong_positive = [
        'шедевр', 'блестящий', 'гениальный', 'восхитительный', 'идеальный', 'безупречный',
        'совершенный', 'незабываемый', 'выдающийся', 'потрясающий', 'невероятный'
    ]
    
    # Проверка СИЛЬНЫХ негативных индикаторов
    strong_neg_count = sum(1 for word in strong_negative if word in text_lower)
    if strong_neg_count >= 1:
        confidence = min(0.85 + (strong_neg_count * 0.03), 0.98)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # Проверка СИЛЬНЫХ позитивных индикаторов
    strong_pos_count = sum(1 for word in strong_positive if word in text_lower)
    if strong_pos_count >= 1:
        confidence = min(0.85 + (strong_pos_count * 0.03), 0.98)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # Если ML доступен - используем его для сложных случаев
    if ML_AVAILABLE and len(text_lower.split()) >= 3:
        try:
            ml_result = ml_analyze_sentiment(text)
            # Если ML уверен - доверяем ему
            if ml_result[1] > 0.7:
                return ml_result
        except Exception as e:
            print(f"⚠️ ML анализ не сработал: {e}")
    
    # Fallback на улучшенные правила
    return advanced_rule_based_analyzer(text)

def advanced_rule_based_analyzer(text):
    """Улучшенный rule-based анализатор с контекстным анализом"""
    text_lower = text.lower()
    
    # Расширенные словари с весами
    positive_words = {
        'отличный': 2, 'великолепный': 2, 'прекрасный': 2, 'супер': 1, 'классный': 1,
        'шикарный': 2, 'восхитительный': 2, 'превосходно': 2, 'замечательный': 1,
        'люблю': 2, 'обожаю': 2, 'рекомендую': 2, 'советую': 2, 'нравится': 1,
        'восторг': 2, 'удовольствие': 1, 'талантливый': 1, 'мастерство': 2,
        'захватывающий': 2, 'трогательный': 1, 'вдохновляющий': 2, 'глубокий': 1,
        'профессиональный': 1, 'качественный': 1, 'динамичный': 1, 'интересный': 1
    }
    
    negative_words = {
        'ужасный': 2, 'плохой': 1, 'скучный': 2, 'отвратительный': 2, 'кошмар': 2,
        'разочарование': 2, 'разочаровал': 2, 'не рекомендую': 3, 'не советую': 3,
        'скучновато': 1, 'затянуто': 1, 'предсказуемо': 1, 'слабый': 1, 'слабая': 1,
        'не стоит': 2, 'жалко времени': 3, 'жалко денег': 2, 'ожидал большего': 2,
        'неудачный': 1, 'провал': 2, 'скучно': 2, 'плохо': 1, 'неинтересно': 1
    }
    
    # Контекстные фразы (высокий вес)
    positive_phrases = [
        'на одном дыхании', 'актерская игра на высоте', 'смотрел на одном дыхании',
        'лучший фильм года', 'пересматривал несколько раз', 'остался под впечатлением',
        'цепляет с первых минут', 'не отпускает до конца', 'операторская работа выше всяких похвал'
    ]
    
    negative_phrases = [
        'зря потратил время', 'сюжетные дыры', 'можно было сократить',
        'персонажи картонные', 'диалоги неестественные', 'спецэффекты выглядят дешево',
        'концовка испортила весь фильм', 'ожидал большего но разочаровался'
    ]
    
    # Подсчет очков
    positive_score = 0
    negative_score = 0
    
    # Слова
    for word, weight in positive_words.items():
        if word in text_lower:
            positive_score += weight
    
    for word, weight in negative_words.items():
        if word in text_lower:
            negative_score += weight
    
    # Фразы (высокий вес)
    for phrase in positive_phrases:
        if phrase in text_lower:
            positive_score += 3
    
    for phrase in negative_phrases:
        if phrase in text_lower:
            negative_score += 3
    
    # Нейтральные индикаторы
    neutral_words = ['средне', 'нормально', 'обычно', 'стандартно', 'ничего особенного', 'так себе']
    neutral_count = sum(1 for word in neutral_words if word in text_lower)
    
    # Логика принятия решения
    total_score = positive_score - negative_score
    
    if neutral_count >= 2 and abs(total_score) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    if total_score > 5:
        confidence = min(0.75 + (total_score * 0.04), 0.95)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    elif total_score < -5:
        confidence = min(0.75 + (abs(total_score) * 0.04), 0.95)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    elif total_score > 2:
        confidence = 0.65 + (total_score * 0.05)
        return "ПОЗИТИВНЫЙ", confidence, "🙂"
    elif total_score < -2:
        confidence = 0.65 + (abs(total_score) * 0.05)
        return "НЕГАТИВНЫЙ", confidence, "😐"
    else:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"

def ml_analyze_sentiment(text):
    """ML анализ с автоматическим обучением"""
    try:
        # Пробуем загрузить существующую модель
        try:
            model = joblib.load('model/sentiment_model.pkl')
            vectorizer = joblib.load('model/vectorizer.pkl')
            print("✅ ML модель загружена из файла")
        except:
            # Если модели нет - создаем улучшенную
            print("🔄 Создаем ML модель...")
            model, vectorizer = create_enhanced_ml_model()
        
        # Анализ
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        confidence = probability[prediction]
        
        if prediction == 1:
            return "ПОЗИТИВНЫЙ", confidence, "😊"
        else:
            return "НЕГАТИВНЫЙ", confidence, "😠"
            
    except Exception as e:
        print(f"❌ ML ошибка: {e}")
        raise

def create_enhanced_ml_model():
    """Создает улучшенную ML модель на лету"""
    # Расширенный датасет для обучения
    positive_texts = [
        "Фильм просто великолепен! Актеры играют превосходно.",
        "Отличный фильм! Смотрел на одном дыхании.",
        "Шедевр! Лучшее что я видел за последнее время.",
        "Прекрасная операторская работа и глубокая драма.",
        "Восхитительная актерская игра, просто вау!",
        "Сюжет захватывающий, не оторваться.",
        "Глубокий и философский фильм.",
        "Напряженный триллер с отличной атмосферой.",
        "Трогательная история о любви и преданности.",
        "Идеальный фильм для вечернего просмотра.",
        "Отличный каст! Все актеры подобраны идеально.",
        "Фильм цепляет с первых минут.",
        "Остался под большим впечатлением.",
        "Рекомендую всем к просмотру!",
        "Пересматривал уже несколько раз.",
        "Сюжет с неожиданными поворотами.",
        "Эмоциональная глубина поражает.",
        "Герои вызывают симпатию с первых минут.",
        "Динамичный сюжет, нет скучных моментов.",
        "Берет за душу, не оставляет равнодушным."
    ]
    
    negative_texts = [
        "Ужасный фильм! Полное разочарование.",
        "Скучно и предсказуемо, не рекомендую.",
        "Зря потратил время на этот фильм.",
        "Плохая актерская игра и слабый сценарий.",
        "Сюжетные дыры видны невооруженным глазом.",
        "Затянуто и скучновато.",
        "Ожидал большего, но разочаровался.",
        "Диалоги звучат неестественно.",
        "Спецэффекты выглядят дешево.",
        "Концовка испортила весь фильм.",
        "Персонажи картонные, невозможно сопереживать.",
        "Слишком много клише и штампов.",
        "Темп неравномерный, то быстро то медленно.",
        "Музыка не подходит к сценам.",
        "Актеры явно не подходят для ролей.",
        "Сюжет нелогичен, персонажи глупые.",
        "Слишком мрачно и депрессивно.",
        "Комедийные моменты неуместны.",
        "Заумно и непонятно.",
        "Слишком много насилия без смысла.",
        "Фильм хуйня! Полное разочарование.",
        "Полное говно! Никому не советую.",
        "Дерьмо собачье! Зря потратил время.",
        "Отстой полный! Лучше бы поспал.",
        "Пиздец какой плохой фильм!"
    ]
    
    # Создаем датасет
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        stop_words=None,
        lowercase=True
    )
    
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    model = LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    
    # Сохраняем модель
    joblib.dump(model, 'model/sentiment_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    accuracy = model.score(X, y)
    print(f"✅ ML модель создана! Точность: {accuracy:.3f}")
    
    return model, vectorizer

# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def analyze_russian_review(russian_text):
    """Анализ одного отзыва с гибридной системой"""
    try:
        sentiment, confidence, emotion = hybrid_sentiment_analyzer(russian_text)
        
        analyzer_type = "ML модель" if ML_AVAILABLE and len(russian_text.split()) >= 3 else "Правила"
        print(f"🔍 {analyzer_type}: '{russian_text[:60]}...' -> {sentiment} ({confidence:.2f})")
        
        return {
            'original_text': russian_text,
            'translated_text': f'Анализ выполнен {analyzer_type.lower()}',
            'sentiment_ru': sentiment,
            'confidence': float(confidence),
            'emotion': emotion
        }
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return {
            'original_text': russian_text,
            'translated_text': f'Ошибка: {str(e)}',
            'sentiment_ru': 'ОШИБКА',
            'confidence': 0.0,
            'emotion': '❌'
        }

def analyze_batch_reviews(texts):
    """Анализ пакета отзывов"""
    results = []
    statistics = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'errors': 0,
        'avg_confidence': 0
    }
    
    for text in texts:
        if text.strip():
            result = analyze_russian_review(text.strip())
            results.append(result)
            
            statistics['total'] += 1
            if result['sentiment_ru'] == 'ПОЗИТИВНЫЙ':
                statistics['positive'] += 1
            elif result['sentiment_ru'] == 'НЕГАТИВНЫЙ':
                statistics['negative'] += 1
            elif result['sentiment_ru'] == 'НЕОПРЕДЕЛЕНО':
                statistics['neutral'] += 1
            else:
                statistics['errors'] += 1
    
    if results:
        valid_confidences = [r['confidence'] for r in results if r['sentiment_ru'] != 'ОШИБКА']
        if valid_confidences:
            statistics['avg_confidence'] = sum(valid_confidences) / len(valid_confidences)
    
    return results, statistics

def create_text_statistics(statistics):
    """Создает текстовое представление статистики"""
    total = statistics['total']
    if total == 0:
        return "Нет данных для отображения"
    
    positive_percent = (statistics['positive'] / total) * 100 if total > 0 else 0
    negative_percent = (statistics['negative'] / total) * 100 if total > 0 else 0
    neutral_percent = (statistics['neutral'] / total) * 100 if total > 0 else 0
    
    text_stats = f"""
📊 СТАТИСТИКА АНАЛИЗА (ГИБРИДНАЯ СИСТЕМА):

✅ Позитивные: {statistics['positive']} ({positive_percent:.1f}%)
❌ Негативные: {statistics['negative']} ({negative_percent:.1f}%) 
😐 Нейтральные: {statistics['neutral']} ({neutral_percent:.1f}%)
⚠️ Ошибки: {statistics['errors']}

📈 Всего отзывов: {total}
🎯 Средняя уверенность: {statistics['avg_confidence']:.1%}
"""
    
    return text_stats

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review_text = data['review'].strip()
        
        if not review_text:
            return jsonify({'success': False, 'error': 'Review is empty'}), 400
        
        result = analyze_russian_review(review_text)
        
        return jsonify({
            'success': True,
            'original_text': result['original_text'],
            'translated_text': result['translated_text'],
            'sentiment_ru': result['sentiment_ru'],
            'confidence': result['confidence'],
            'emotion': result['emotion']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        print("📨 Начало обработки пакетного запроса...")
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.txt'):
            return jsonify({'success': False, 'error': 'Only .txt files allowed'}), 400
        
        # Читаем файл
        content = file.read().decode('utf-8')
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"📊 Анализируем {len(texts)} отзывов с гибридной системой...")
        
        if len(texts) == 0:
            return jsonify({'success': False, 'error': 'File is empty'}), 400
        
        # Анализируем
        results, statistics = analyze_batch_reviews(texts)
        text_stats = create_text_statistics(statistics)
        
        print(f"✅ Гибридный анализ завершен. Обработано: {len(results)} отзывов")
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': statistics,
            'text_stats': text_stats,
            'processed_count': len(results)
        })
            
    except Exception as e:
        print(f"❌ Ошибка в analyze_batch: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/retrain_ml')
def retrain_ml():
    """Переобучение ML модели"""
    if not ML_AVAILABLE:
        return jsonify({'success': False, 'error': 'ML не доступен'})
    
    try:
        global ml_model, ml_vectorizer
        ml_model, ml_vectorizer = create_enhanced_ml_model()
        return jsonify({
            'success': True,
            'message': 'ML модель переобучена!'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test')
def test():
    ml_status = "доступна" if ML_AVAILABLE else "не доступна"
    return jsonify({
        'message': 'Сервер работает с гибридной системой анализа!', 
        'status': 'OK',
        'analyzer_type': 'Гибридный (Правила + ML)',
        'ml_status': ml_status,
        'version': '3.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
