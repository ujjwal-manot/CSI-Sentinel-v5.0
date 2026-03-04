#include "csi_collector.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <string.h>
#include <math.h>

static csi_data_callback_t s_callback = NULL;
static uint8_t s_device_id = 0;
static uint16_t s_sequence = 0;
static uint32_t s_packet_count = 0;
static QueueHandle_t s_csi_queue = NULL;
static TaskHandle_t s_process_task = NULL;
static volatile bool s_running = false;

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (!s_running || !info || !info->buf) return;

    csi_packet_t packet;
    packet.magic = 0xC510;
    packet.version = 0x01;
    packet.device_id = s_device_id;
    packet.timestamp_us = (uint32_t)esp_timer_get_time();
    packet.sequence = s_sequence++;
    packet.rssi = info->rx_ctrl.rssi;
    packet.noise_floor = info->rx_ctrl.noise_floor;
    packet.channel = info->rx_ctrl.channel;
    packet.secondary_channel = info->rx_ctrl.secondary_channel;

    int len = info->len;
    if (len > 128) len = 128;
    packet.num_subcarriers = len / 2;

    memcpy(packet.csi_data, info->buf, len);

    if (s_csi_queue) {
        xQueueSendFromISR(s_csi_queue, &packet, NULL);
    }
}

static void csi_process_task(void *arg) {
    csi_packet_t packet;

    while (s_running) {
        if (xQueueReceive(s_csi_queue, &packet, pdMS_TO_TICKS(100)) == pdTRUE) {
            s_packet_count++;
            if (s_callback) {
                s_callback(&packet);
            }
        }
    }

    vTaskDelete(NULL);
}

esp_err_t csi_collector_init(uint8_t device_id) {
    s_device_id = device_id;
    s_sequence = 0;
    s_packet_count = 0;

    s_csi_queue = xQueueCreate(64, sizeof(csi_packet_t));
    if (!s_csi_queue) return ESP_ERR_NO_MEM;

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };

    esp_err_t ret = esp_wifi_set_csi_config(&csi_config);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL);
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

esp_err_t csi_collector_start(void) {
    s_running = true;

    esp_err_t ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) {
        s_running = false;
        return ret;
    }

    BaseType_t task_ret = xTaskCreatePinnedToCore(
        csi_process_task,
        "csi_process",
        4096,
        NULL,
        5,
        &s_process_task,
        1
    );

    if (task_ret != pdPASS) {
        s_running = false;
        esp_wifi_set_csi(false);
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

void csi_collector_stop(void) {
    s_running = false;
    esp_wifi_set_csi(false);

    if (s_process_task) {
        vTaskDelay(pdMS_TO_TICKS(200));
        s_process_task = NULL;
    }
}

void csi_collector_register_callback(csi_data_callback_t callback) {
    s_callback = callback;
}

uint32_t csi_collector_get_packet_count(void) {
    return s_packet_count;
}
