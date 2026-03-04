#ifndef CSI_COLLECTOR_H
#define CSI_COLLECTOR_H

#include "esp_err.h"
#include "csi_config.h"

typedef void (*csi_data_callback_t)(const csi_packet_t *packet);

esp_err_t csi_collector_init(uint8_t device_id);
esp_err_t csi_collector_start(void);
void csi_collector_stop(void);
void csi_collector_register_callback(csi_data_callback_t callback);
uint32_t csi_collector_get_packet_count(void);

#endif
