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
    """Умный анализатор без сложных зависимостей"""
    text_lower = text.lower().strip()
    
    if not text_lower or len(text_lower) < 3:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # КЛЮЧЕВЫЕ СЛОВА И ФРАЗЫ
    positive_indicators = [
        # Сильные позитивные
        'шедевр', 'блестящий', 'гениальный', 'восхитительный', 'идеальный',
        'безупречный', 'незабываемый', 'потрясающий', 'невероятный', 'фантастический',
        
        # Обычные позитивные
        'отличный', 'великолепный', 'прекрасный', 'супер', 'классный', 'шикарный',
        'превосходно', 'замечательный', 'люблю', 'обожаю', 'рекомендую', 'советую',
        'нравится', 'восторг', 'талантливый', 'мастерство', 'захватывающий',
        'трогательный', 'вдохновляющий', 'глубокий', 'профессиональный', 'качественный',
        
        # Контекстные фразы
        'на одном дыхании', 'актерская игра на высоте', 'лучший фильм года',
        'пересматривал несколько раз', 'остался под впечатлением', 'цепляет с первых минут',
        'не отпускает до конца', 'операторская работа выше всяких похвал', 'берет за душу'
    ]
    
    negative_indicators = [
        # Сильные негативные (матерные)
        'хуйня', 'говно', 'пиздец', 'дерьмо', 'отстой', 'мудак', 'гандон',
        'ублюдок', 'пидорас', 'залупа', 'поебень', 'говнище', 'ебанько',
        
        # Обычные негативные
        'отвратительно', 'отвратительный', 'ужасный', 'кошмар', 'провал',
        'не рекомендую', 'не советую', 'зря потратил', 'разочарование', 'разочаровал',
        'скучно', 'скучный', 'затянуто', 'предсказуемо', 'слабый', 'слабая',
        'плохой', 'плохая', 'не стоит', 'жалко времени', 'жалко денег',
        'ожидал большего', 'неудачный', 'неинтересно', 'раздражает', 'бесит', 'ненавижу',
        
        # Контекстные фразы
        'зря потратил время', 'сюжетные дыры', 'персонажи картонные',
        'диалоги неестественные', 'спецэффекты выглядят дешево', 'концовка испортила весь фильм',
        'первая половина интересная вторая разочаровала', 'сюжет нелогичен',
        'персонажи глупые', 'актеры не подходят для ролей', 'режиссер не справился'
    ]
    
    neutral_indicators = [
        'средне', 'нормально', 'обычно', 'стандартно', 'ничего особенного',
        'так себе', 'посредственно', 'рядовой', 'обыкновенный', 'типично'
    ]
    
    # ПОДСЧЕТ СОВПАДЕНИЙ
    positive_matches = []
    negative_matches = []
    neutral_matches = []
    
    for indicator in positive_indicators:
        if indicator in text_lower:
            positive_matches.append(indicator)
    
    for indicator in negative_indicators:
        if indicator in text_lower:
            negative_matches.append(indicator)
    
    for indicator in neutral_indicators:
        if indicator in text_lower:
            neutral_matches.append(indicator)
    
    # СИЛЬНЫЕ ИНДИКАТОРЫ (матерные слова)
    strong_negative_words = ['хуйня', 'говно', 'пиздец', 'дерьмо', 'мудак', 'гандон']
    strong_positive_words = ['шедевр', 'блестящий', 'гениальный', 'восхитительный']
    
    for word in strong_negative_words:
        if word in text_lower:
            return "НЕГАТИВНЫЙ", 0.95, "😠"
    
    for word in strong_positive_words:
        if word in text_lower:
            return "ПОЗИТИВНЫЙ", 0.95, "😊"
    
    # ЛОГИКА ПРИНЯТИЯ РЕШЕНИЯ
    pos_count = len(positive_matches)
    neg_count = len(negative_matches)
    neutral_count = len(neutral_matches)
    
    # Явно позитивные
    if pos_count >= 3 and neg_count == 0:
        confidence = min(0.8 + (pos_count * 0.05), 0.95)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # Явно негативные
    if neg_count >= 3 and pos_count == 0:
        confidence = min(0.8 + (neg_count * 0.05), 0.95)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # Умеренно позитивные
    if pos_count > neg_count and pos_count >= 2:
        confidence = min(0.7 + (pos_count * 0.05), 0.85)
        return "ПОЗИТИВНЫЙ", confidence, "🙂"
    
    # Умеренно негативные
    if neg_count > pos_count and neg_count >= 2:
        confidence = min(0.7 + (neg_count * 0.05), 0.85)
        return "НЕГАТИВНЫЙ", confidence, "😐"
    
    # Нейтральные
    if neutral_count >= 2 and pos_count == 0 and neg_count == 0:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # Смешанные эмоции
    if pos_count > 0 and neg_count > 0:
        if pos_count > neg_count:
            return "ПОЗИТИВНЫЙ", 0.6, "🙂"
        else:
            return "НЕГАТИВНЫЙ", 0.6, "😐"
    
    # Слабые сигналы
    if pos_count == 1 and neg_count == 0:
        return "ПОЗИТИВНЫЙ", 0.65, "🙂"
    
    if neg_count == 1 and pos_count == 0:
        return "НЕГАТИВНЫЙ", 0.65, "😐"
    
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
        'version': '5.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
