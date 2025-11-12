from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== УЛУЧШЕННЫЙ RULE-BASED АНАЛИЗАТОР ====================

def advanced_sentiment_analyzer(text):
    """Продвинутый rule-based анализатор с контекстным анализом"""
    text_lower = text.lower().strip()
    
    # Быстрая проверка пустого текста
    if not text_lower or len(text_lower) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # СИЛЬНЫЕ НЕГАТИВНЫЕ ИНДИКАТОРЫ (ВЫСОКИЙ ПРИОРИТЕТ)
    strong_negative = {
        'хуйня': 3, 'говно': 3, 'пиздец': 3, 'дерьмо': 3, 'отстой': 2, 
        'отвратительно': 2, 'отвратительный': 2, 'ужасный': 2, 'кошмар': 2,
        'провал': 2, 'мудак': 3, 'гандон': 3, 'ублюдок': 3, 'пидорас': 3,
        'залупа': 3, 'поебень': 3, 'говнище': 3, 'ебанько': 3
    }
    
    # СИЛЬНЫЕ ПОЗИТИВНЫЕ ИНДИКАТОРЫ (ВЫСОКИЙ ПРИОРИТЕТ)
    strong_positive = {
        'шедевр': 3, 'блестящий': 2, 'гениальный': 2, 'восхитительный': 2,
        'идеальный': 2, 'безупречный': 2, 'совершенный': 2, 'незабываемый': 2,
        'выдающийся': 2, 'потрясающий': 2, 'невероятный': 2, 'фантастический': 2
    }
    
    # ОБЫЧНЫЕ СЛОВА
    positive_words = {
        'отличный': 2, 'великолепный': 2, 'прекрасный': 2, 'супер': 1, 'классный': 1,
        'шикарный': 2, 'превосходно': 2, 'замечательный': 1, 'люблю': 2, 'обожаю': 2,
        'рекомендую': 2, 'советую': 2, 'нравится': 1, 'восторг': 2, 'удовольствие': 1,
        'талантливый': 1, 'мастерство': 2, 'захватывающий': 2, 'трогательный': 1,
        'вдохновляющий': 2, 'глубокий': 1, 'профессиональный': 1, 'качественный': 1,
        'динамичный': 1, 'интересный': 1, 'увлекательный': 1, 'очаровательный': 1
    }
    
    negative_words = {
        'не рекомендую': 3, 'не советую': 3, 'зря потратил': 3, 'разочарование': 2,
        'разочаровал': 2, 'скучно': 2, 'скучный': 2, 'затянуто': 1, 'предсказуемо': 1,
        'слабый': 1, 'слабая': 1, 'плохой': 1, 'плохая': 1, 'не стоит': 2, 
        'жалко времени': 3, 'жалко денег': 2, 'ожидал большего': 2, 'неудачный': 1,
        'скучновато': 1, 'неинтересно': 1, 'раздражает': 1, 'бесит': 2, 'ненавижу': 2
    }
    
    # КОНТЕКСТНЫЕ ФРАЗЫ (ВЫСОКИЙ ВЕС)
    positive_phrases = {
        'на одном дыхании': 3, 'актерская игра на высоте': 3, 'смотрел на одном дыхании': 3,
        'лучший фильм года': 3, 'пересматривал несколько раз': 3, 'остался под впечатлением': 2,
        'цепляет с первых минут': 2, 'не отпускает до конца': 2, 'операторская работа выше всяких похвал': 2,
        'глубокая психологическая драма': 2, 'эмоциональная глубина': 2, 'берет за душу': 2,
        'не оставляет равнодушным': 2, 'полное погружение': 2, 'шедевр кинематографа': 3
    }
    
    negative_phrases = {
        'зря потратил время': 3, 'сюжетные дыры': 2, 'можно было сократить': 1,
        'персонажи картонные': 2, 'диалоги неестественные': 2, 'спецэффекты выглядят дешево': 2,
        'концовка испортила весь фильм': 3, 'ожидал большего но разочаровался': 2,
        'первая половина интересная вторая разочаровала': 2, 'сюжет нелогичен': 2,
        'персонажи глупые': 2, 'актеры не подходят для ролей': 2, 'режиссер не справился': 2
    }
    
    # НЕЙТРАЛЬНЫЕ СЛОВА
    neutral_words = [
        'средне', 'нормально', 'обычно', 'стандартно', 'ничего особенного', 
        'так себе', 'посредственно', 'рядовой', 'обыкновенный', 'типично'
    ]
    
    # ПОДСЧЕТ ОЧКОВ
    positive_score = 0
    negative_score = 0
    
    # Сильные негативные (максимальный вес)
    for word, weight in strong_negative.items():
        if word in text_lower:
            negative_score += weight
    
    # Сильные позитивные (максимальный вес)
    for word, weight in strong_positive.items():
        if word in text_lower:
            positive_score += weight
    
    # Обычные слова
    for word, weight in positive_words.items():
        if word in text_lower:
            positive_score += weight
    
    for word, weight in negative_words.items():
        if word in text_lower:
            negative_score += weight
    
    # Контекстные фразы (очень высокий вес)
    for phrase, weight in positive_phrases.items():
        if phrase in text_lower:
            positive_score += weight
    
    for phrase, weight in negative_phrases.items():
        if phrase in text_lower:
            negative_score += weight
    
    # Нейтральные индикаторы
    neutral_count = sum(1 for word in neutral_words if word in text_lower)
    
    # АНАЛИЗ УСИЛИТЕЛЕЙ И ОТРИЦАНИЙ
    words = text_lower.split()
    for i, word in enumerate(words):
        # Усилители
        if word in ['очень', 'крайне', 'невероятно', 'абсолютно', 'совершенно']:
            if i + 1 < len(words):
                next_word = words[i + 1]
                if any(pos in next_word for pos in positive_words):
                    positive_score += 1
                elif any(neg in next_word for neg in negative_words):
                    negative_score += 1
        
        # Отрицания
        elif word in ['не', 'ни', 'без']:
            if i + 1 < len(words):
                next_word = words[i + 1]
                if any(pos in next_word for pos in positive_words):
                    negative_score += 2  # "не отличный" → негатив
                elif any(neg in next_word for neg in negative_words):
                    positive_score += 2  # "не плохой" → позитив
    
    # ЛОГИКА ПРИНЯТИЯ РЕШЕНИЯ
    total_score = positive_score - negative_score
    
    # Нейтральные случаи
    if neutral_count >= 2 and abs(total_score) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # Явно позитивные
    if total_score > 8:
        confidence = min(0.9 + (total_score * 0.01), 0.98)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # Явно негативные
    if total_score < -8:
        confidence = min(0.9 + (abs(total_score) * 0.01), 0.98)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # Умеренно позитивные
    if total_score > 4:
        confidence = min(0.75 + (total_score * 0.03), 0.88)
        return "ПОЗИТИВНЫЙ", confidence, "🙂"
    
    # Умеренно негативные
    if total_score < -4:
        confidence = min(0.75 + (abs(total_score) * 0.03), 0.88)
        return "НЕГАТИВНЫЙ", confidence, "😐"
    
    # Слабые сигналы
    if total_score > 0:
        return "ПОЗИТИВНЫЙ", 0.6, "🙂"
    elif total_score < 0:
        return "НЕГАТИВНЫЙ", 0.6, "😐"
    
    # По умолчанию
    return "НЕОПРЕДЕЛЕНО", 0.5, "😐"

# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def analyze_russian_review(russian_text):
    """Анализ одного отзыва"""
    try:
        sentiment, confidence, emotion = advanced_sentiment_analyzer(russian_text)
        
        print(f"🔍 Анализ: '{russian_text[:60]}...' -> {sentiment} ({confidence:.2f})")
        
        return {
            'original_text': russian_text,
            'translated_text': 'Анализ выполнен улучшенной системой',
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
📊 СТАТИСТИКА АНАЛИЗА (УЛУЧШЕННАЯ СИСТЕМА):

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
        
        print(f"📊 Анализируем {len(texts)} отзывов...")
        
        if len(texts) == 0:
            return jsonify({'success': False, 'error': 'File is empty'}), 400
        
        # Анализируем
        results, statistics = analyze_batch_reviews(texts)
        text_stats = create_text_statistics(statistics)
        
        print(f"✅ Анализ завершен. Обработано: {len(results)} отзывов")
        
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

@app.route('/test')
def test():
    return jsonify({
        'message': 'Сервер работает с улучшенной системой анализа!', 
        'status': 'OK',
        'analyzer_type': 'Advanced Rule-Based',
        'features': 'Контекстный анализ, веса слов, усилители',
        'version': '4.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
