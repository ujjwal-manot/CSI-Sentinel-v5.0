#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include "esp_err.h"

esp_err_t wifi_manager_init_softap(void);
esp_err_t wifi_manager_init_station(void);
esp_err_t wifi_manager_start(void);
void wifi_manager_stop(void);
bool wifi_manager_is_connected(void);

#endif
