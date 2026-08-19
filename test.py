import time 
import  os 
import webbrowser 


def start_beauty_ai_timer (minutes=60) :
    print(f"🚀 تایمر شگفتی‌سازی BeautyAI برای {minutes} دقیقه دیگر فعال شد.")
    print(f"⏰ سر ساعت، بورد تسک‌ها و آلارم برای بررسی فاز ۲ بک‌آند و فرانت فعال می‌شود...")
    
    time.sleep(minutes * 60)
    print("\n🚨 TIME'S UP! زمان بررسی تسک‌های T-B01 و T-F01 رسید! 🚨")
    
    for _ in range(5) :
        os.system('echo \a')
        time.sleep(0.5)

if __name__  == "__main__" :
    start_beauty_ai_timer(120)