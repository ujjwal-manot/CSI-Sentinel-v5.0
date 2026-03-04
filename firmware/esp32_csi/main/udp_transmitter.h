#ifndef UDP_TRANSMITTER_H
#define UDP_TRANSMITTER_H

#include "esp_err.h"
#include "csi_config.h"

esp_err_t udp_transmitter_init(const char *server_ip, uint16_t server_port);
esp_err_t udp_transmitter_send(const csi_packet_t *packet);
void udp_transmitter_close(void);
uint32_t udp_transmitter_get_bytes_sent(void);

#endif
