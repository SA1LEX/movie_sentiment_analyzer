from flask import Flask, render_template, request, jsonify
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== УПРОЩЕННЫЙ АНАЛИЗАТОР ====================

def advanced_sentiment_analyzer(text):
    """Улучшенный rule-based анализатор тональности"""
    text_lower = text.lower()
    
    # Сильно негативные слова (ВЫСОКИЙ ПРИОРИТЕТ)
    strong_negative = [
        'хуйня', 'говно', 'пиздец', 'дерьмо', 'отстой', 'мудак', 'гандон', 
        'ублюдок', 'пидорас', 'залупа', 'поебень', 'говнище', 'ебанько',
        'отвратительно', 'отвратительный', 'ужасный', 'кошмар', 'провал'
    ]
    
    # Умеренно негативные фразы
    moderate_negative = [
        'не рекомендую', 'не советую', 'зря потратил', 'разочарование',
        'скучно', 'скучный', 'затянуто', 'предсказуемо', 'слабый', 'слабая',
        'плохой', 'плохая', 'не стоит', 'жалко времени', 'жалко денег',
        'скучновато', 'разочаровал', 'ожидал большего'
    ]
    
    # Позитивные слова
    positive_words = [
        'отличный', 'великолепный', 'прекрасный', 'супер', 'классный', 
        'шедевр', 'люблю', 'рекомендую', 'советую', 'нравится', 'восторг',
        'восхитительный', 'превосходно', 'идеальный', 'блестящий', 'гениальный',
        'талантливый', 'мастерство', 'безупречный', 'совершенный'
    ]
    
    # Нейтральные слова
    neutral_words = [
        'средне', 'нормально', 'обычно', 'стандартно', 'типично',
        'ничего', 'так себе', 'посредственно', 'рядовой', 'обыкновенный'
    ]
    
    # Подсчет очков
    strong_neg_count = sum(1 for word in strong_negative if word in text_lower)
    moderate_neg_count = sum(1 for word in moderate_negative if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neutral_count = sum(1 for word in neutral_words if word in text_lower)
    
    # ПРАВИЛА КЛАССИФИКАЦИИ:
    
    # 1. Сильно негативные - сразу негатив
    if strong_neg_count >= 1:
        confidence = min(0.8 + (strong_neg_count * 0.05), 0.95)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # 2. Умеренно негативные
    if moderate_neg_count >= 2:
        confidence = min(0.7 + (moderate_neg_count * 0.05), 0.85)
        return "НЕГАТИВНЫЙ", confidence, "😠"
    
    # 3. Позитивные
    if pos_count >= 2:
        confidence = min(0.7 + (pos_count * 0.05), 0.9)
        return "ПОЗИТИВНЫЙ", confidence, "😊"
    
    # 4. Нейтральные
    if neutral_count >= 1 and strong_neg_count == 0 and moderate_neg_count == 0 and pos_count == 0:
        return "НЕОПРЕДЕЛЕНО", 0.5, "😐"
    
    # 5. Смешанные эмоции
    if pos_count > 0 and (moderate_neg_count > 0 or strong_neg_count > 0):
        if pos_count > (moderate_neg_count + strong_neg_count):
            return "ПОЗИТИВНЫЙ", 0.6, "🙂"
        else:
            return "НЕГАТИВНЫЙ", 0.6, "😐"
    
    # 6. По умолчанию - неопределенный
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
        
        print(f"✅ Анализ завершен. Обработано: {len(results)} отзывов")
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': statistics,
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
        'version': '2.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
