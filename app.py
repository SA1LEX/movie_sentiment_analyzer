from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== ПРОСТОЙ И ЭФФЕКТИВНЫЙ АНАЛИЗАТОР ====================

def smart_sentiment_analyzer(text):
    """Умный rule-based анализатор без зависимостей"""
    text_lower = text.lower().strip()
    
    if not text_lower or len(text_lower) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # РАСШИРЕННЫЕ СЛОВАРИ
    positive_words = [
        # Сильные позитивные
        'великолепен', 'великолепный', 'великолепно', 'шедевр', 'блестящий',
        'гениальный', 'восхитительный', 'идеальный', 'безупречный', 'незабываемый',
        'потрясающий', 'невероятный', 'фантастический', 'превосходно', 'совершенный',
        'выдающийся', 'прекрасный', 'очаровательный', 'люблю', 'обожаю',
        
        # Обычные позитивные
        'отличный', 'супер', 'классный', 'шикарный', 'замечательный', 
        'рекомендую', 'советую', 'нравится', 'талантливый', 'мастерство',
        'захватывающий', 'трогательный', 'вдохновляющий', 'глубокий', 
        'профессиональный', 'качественный', 'интересный', 'увлекательный',
        'динамичный', 'симпатичный', 'обаятельный', 'умный', 'оригинальный',
        'свежий', 'новаторский', 'хороший', 'забавный', 'смешной', 'юморной',
        'сильный', 'мощный', 'эпичный', 'красивый', 'эстетичный', 'стильный'
    ]
    
    negative_words = [
        # Сильные негативные (матерные)
        'хуйня', 'говно', 'пиздец', 'дерьмо', 'отстой', 'мудак', 'гандон',
        'ублюдок', 'пидорас', 'залупа', 'поебень', 'говнище', 'ебанько',
        
        # Сильные негативные (обычные)
        'отвратительно', 'отвратительный', 'ужасный', 'кошмар', 'провал', 'разочарование',
        'ненавижу', 'бесит', 'раздражает', 'омерзительно', 'гадость', 'мерзость',
        
        # Обычные негативные
        'не рекомендую', 'не советую', 'зря потратил', 'разочаровал', 'скучно', 'скучный',
        'затянуто', 'предсказуемо', 'слабый', 'слабая', 'плохой', 'плохая', 'не стоит',
        'жалко времени', 'жалко денег', 'ожидал большего', 'неудачный', 'неинтересно',
        'банальный', 'шаблонный', 'клишированный', 'примитивный', 'нелогичный',
        'глупый', 'абсурдный', 'нереалистичный', 'фальшивый', 'неестественный',
        'картонный', 'бездушный', 'безвкусный', 'дешевый', 'кустарный'
    ]
    
    # КОНТЕКСТНЫЕ ФРАЗЫ (высокий вес)
    positive_phrases = [
        'на одном дыхании', 'актерская игра на высоте', 'лучший фильм года',
        'пересматривал несколько раз', 'остался под впечатлением', 
        'цепляет с первых минут', 'не отпускает до конца', 
        'операторская работа выше всяких похвал', 'берет за душу',
        'не оставляет равнодушным', 'глубокий смысл', 'философский подтекст'
    ]
    
    negative_phrases = [
        'зря потратил время', 'сюжетные дыры', 'персонажи картонные',
        'диалоги неестественные', 'спецэффекты выглядят дешево', 
        'концовка испортила весь фильм', 'первая половина интересная вторая разочаровала',
        'сюжет нелогичен', 'персонажи глупые', 'актеры не подходят для ролей'
    ]
    
    # ПОДСЧЕТ ОЧКОВ
    positive_score = 0
    negative_score = 0
    
    # СЛОВА
    for word in positive_words:
        if word in text_lower:
            positive_score += 2
    
    for word in negative_words:
        if word in text_lower:
            negative_score += 2
    
    # ФРАЗЫ (высокий вес)
    for phrase in positive_phrases:
        if phrase in text_lower:
            positive_score += 3
    
    for phrase in negative_phrases:
        if phrase in text_lower:
            negative_score += 3
    
    # АНАЛИЗ УСИЛИТЕЛЕЙ
    words = text_lower.split()
    for i, word in enumerate(words):
        # Усилители
        if word in ['очень', 'крайне', 'невероятно', 'абсолютно', 'совершенно', 'просто']:
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
    
    # Явно позитивные
    if total_score >= 3:
        confidence = min(0.8 + (total_score * 0.05), 0.95)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # Явно негативные
    if total_score <= -3:
        confidence = min(0.8 + (abs(total_score) * 0.05), 0.95)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # Слабые сигналы
    if total_score > 0:
        confidence = 0.6 + (total_score * 0.1)
        return "ПОЗИТИВНЫЙ", min(confidence, 0.75), "🙂"
    
    if total_score < 0:
        confidence = 0.6 + (abs(total_score) * 0.1)
        return "НЕГАТИВНЫЙ", min(confidence, 0.75), "😐"
    
    # По умолчанию
    return "НЕОПРЕДЕЛЕНО", 0.5, "😐"

# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def analyze_russian_review(russian_text):
    """Анализ одного отзыва"""
    try:
        sentiment, confidence, emotion = smart_sentiment_analyzer(russian_text)
        
        print(f"🔍 Анализ: '{russian_text[:60]}...' -> {sentiment} ({confidence:.2f})")
        
        return {
            'original_text': russian_text,
            'translated_text': 'Анализ выполнен умной системой',
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
📊 СТАТИСТИКА АНАЛИЗА:

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
        'message': 'Сервер работает с умной системой анализа!', 
        'status': 'OK',
        'analyzer_type': 'Smart Rule-Based',
        'version': '2.0'
    })

@app.route('/test_analyzer')
def test_analyzer():
    """Тестирование анализатора на контрольных примерах"""
    test_cases = [
        "Фильм просто великолепен!",
        "Отличный фильм!",
        "Шедевр!",
        "Фильм хуйня!",
        "Полное говно!",
        "Скучно и предсказуемо",
        "Нормальный фильм"
    ]
    
    results = []
    for text in test_cases:
        sentiment, confidence, emotion = smart_sentiment_analyzer(text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion
        })
    
    return jsonify({'test_results': results})

if __name__ == '__main__':
    print("🚀 Запускаем Flask сервер с умной системой анализа...")
    print("🔗 URL: http://localhost:5000")
    print("🧪 Тест анализатора: http://localhost:5000/test_analyzer")
    app.run(debug=True, port=5000)
