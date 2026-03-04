#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "csi_config.h"
#include "wifi_manager.h"
#include "csi_collector.h"
#include "udp_transmitter.h"

#define LED_GPIO            2
#define DEVICE_ROLE         DEVICE_ROLE_RX
#define DEVICE_ID           1
#define PING_INTERVAL_MS    5000

static const char *TAG = "CSI_SENTINEL";
static volatile uint32_t s_last_packet_time = 0;
static esp_timer_handle_t s_ping_timer = NULL;

static void led_init(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(LED_GPIO, 0);
}

static void led_blink(void) {
    gpio_set_level(LED_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(LED_GPIO, 0);
}

static void csi_data_handler(const csi_packet_t *packet) {
    udp_transmitter_send(packet);
    s_last_packet_time = (uint32_t)(esp_timer_get_time() / 1000);

    static uint32_t blink_counter = 0;
    if (++blink_counter >= 50) {
        led_blink();
        blink_counter = 0;
    }
}

static void ping_timer_callback(void *arg) {
    if (!wifi_manager_is_connected()) {
        return;
    }

    csi_packet_t ping_packet = {
        .magic = 0xC510,
        .version = 0x01,
        .device_id = DEVICE_ID,
        .timestamp_us = (uint32_t)esp_timer_get_time(),
        .sequence = 0xFFFF,
        .rssi = 0,
        .noise_floor = 0,
        .channel = CSI_WIFI_CHANNEL,
        .secondary_channel = 0,
        .num_subcarriers = 0,
    };
    memset(ping_packet.csi_data, 0, sizeof(ping_packet.csi_data));

    udp_transmitter_send(&ping_packet);
}

static void status_task(void *arg) {
    while (1) {
        uint32_t packets = csi_collector_get_packet_count();
        uint32_t bytes = udp_transmitter_get_bytes_sent();

        ESP_LOGI(TAG, "Packets: %lu | Bytes: %lu KB | Connected: %s",
                 packets, bytes / 1024,
                 wifi_manager_is_connected() ? "Yes" : "No");

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

static void tx_beacon_task(void *arg) {
    wifi_config_t cfg;
    esp_wifi_get_config(WIFI_IF_AP, &cfg);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(CSI_SAMPLE_INTERVAL_US / 1000));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "CSI-Sentinel v5.0 Starting...");
    ESP_LOGI(TAG, "Device Role: %s | ID: %d",
             DEVICE_ROLE == DEVICE_ROLE_TX ? "TX" : "RX", DEVICE_ID);

    led_init();

    esp_err_t ret;

    if (DEVICE_ROLE == DEVICE_ROLE_TX) {
        ret = wifi_manager_init_softap();
    } else {
        ret = wifi_manager_init_station();
    }

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "WiFi init failed: %s", esp_err_to_name(ret));
        return;
    }

    ret = wifi_manager_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "WiFi start failed: %s", esp_err_to_name(ret));
        return;
    }

    ESP_LOGI(TAG, "WiFi started successfully");

    if (DEVICE_ROLE == DEVICE_ROLE_RX) {
        vTaskDelay(pdMS_TO_TICKS(3000));

        ret = udp_transmitter_init(UDP_SERVER_IP, UDP_SERVER_PORT);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "UDP init failed");
            return;
        }

        ret = csi_collector_init(DEVICE_ID);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "CSI collector init failed");
            return;
        }

        csi_collector_register_callback(csi_data_handler);

        ret = csi_collector_start();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "CSI collector start failed");
            return;
        }

        const esp_timer_create_args_t ping_timer_args = {
            .callback = ping_timer_callback,
            .name = "ping_timer"
        };
        esp_timer_create(&ping_timer_args, &s_ping_timer);
        esp_timer_start_periodic(s_ping_timer, PING_INTERVAL_MS * 1000);

        ESP_LOGI(TAG, "CSI Collection Active @ %d Hz", CSI_SAMPLE_RATE_HZ);
    } else {
        xTaskCreate(tx_beacon_task, "tx_beacon", 2048, NULL, 5, NULL);
        ESP_LOGI(TAG, "TX Beacon Active");
    }

    xTaskCreate(status_task, "status", 2048, NULL, 3, NULL);

    gpio_set_level(LED_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(500));
    gpio_set_level(LED_GPIO, 0);
}
