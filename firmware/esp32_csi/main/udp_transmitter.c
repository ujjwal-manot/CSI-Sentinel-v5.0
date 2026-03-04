#include "udp_transmitter.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include <string.h>

static int s_socket = -1;
static struct sockaddr_in s_server_addr;
static uint32_t s_bytes_sent = 0;

esp_err_t udp_transmitter_init(const char *server_ip, uint16_t server_port) {
    s_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_socket < 0) {
        return ESP_FAIL;
    }

    int broadcast = 1;
    setsockopt(s_socket, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct timeval timeout = {
        .tv_sec = 0,
        .tv_usec = 100000
    };
    setsockopt(s_socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    memset(&s_server_addr, 0, sizeof(s_server_addr));
    s_server_addr.sin_family = AF_INET;
    s_server_addr.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip, &s_server_addr.sin_addr);

    s_bytes_sent = 0;

    return ESP_OK;
}

esp_err_t udp_transmitter_send(const csi_packet_t *packet) {
    if (s_socket < 0 || !packet) {
        return ESP_ERR_INVALID_STATE;
    }

    int sent = sendto(
        s_socket,
        packet,
        sizeof(csi_packet_t),
        0,
        (struct sockaddr *)&s_server_addr,
        sizeof(s_server_addr)
    );

    if (sent < 0) {
        return ESP_FAIL;
    }

    s_bytes_sent += sent;
    return ESP_OK;
}

void udp_transmitter_close(void) {
    if (s_socket >= 0) {
        close(s_socket);
        s_socket = -1;
    }
}

uint32_t udp_transmitter_get_bytes_sent(void) {
    return s_bytes_sent;
}
