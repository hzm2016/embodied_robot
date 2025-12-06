import redis
import time
import threading



if __name__ == "__main__":
    # Create a Publisher instance
    publisher = Publisher()
    # Create a thread for publishing messages
    publisher_thread = threading.Thread(target=publisher.publish_messages)
    # Start the publisher thread
    publisher_thread.start()
    try:
        # Main loop to update positions
        i = 0
        while True:
            i += 0.001

            # Update endo_start and endo_end dynamically in the main loop
            endo_start = [0, 0, 200]
            endo_end = [100 + i, 0, 200]
            publisher.update_positions(endo_start, endo_end)

            time.sleep(0.5)  # Simulate main loop delay
    except KeyboardInterrupt:
        print("\nStopping the publisher...")