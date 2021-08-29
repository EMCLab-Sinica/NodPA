/*
 *  main.c
 *
 *  Author: Meenchen
 */

#include <Tools/myuart.h>
#include <Tools/dvfs.h>
#include <Tools/ext_fram/extfram.h> // for EXTFRAM_USE_DMA

#include "plat-msp430.h"

/* Standard demo includes, used so the tick hook can exercise some FreeRTOS
functionality in an interrupt. */
#include <driverlib.h>
#include "main.h"

/*-----------------------------------------------------------*/
/*
 * Configure the hardware as necessary.
 */
static void prvSetupHardware( void );
#ifndef EXTFRAM_USE_DMA
static void vApplicationSetupTimerInterrupt( void );
#endif

/*-----------------------------------------------------------*/

int main( void )
{
    /* Configure the hardware ready to run the demo. */
    prvSetupHardware();

    GPIO_setOutputHighOnPin(GPIO_PORT_P1, GPIO_PIN0);

    // XXX: disabled - timer intterupts appear to interfere DMA read for external FRAM
#ifndef EXTFRAM_USE_DMA
    vApplicationSetupTimerInterrupt();
#endif
    IntermittentCNNTest();

	return 0;
}


static void prvSetupHardware( void )
{
    /* Stop Watchdog timer. */
    WDT_A_hold( __MSP430_BASEADDRESS_WDT_A__ );

	/* Set all GPIO pins to output and low. */
	GPIO_setOutputLowOnPin( GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 | GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 | GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15 );
	GPIO_setAsOutputPin( GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 | GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 | GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15 );

	/* Configure P2.0 - UCA0TXD and P2.1 - UCA0RXD. */
	GPIO_setOutputLowOnPin( GPIO_PORT_P2, GPIO_PIN0 );
	GPIO_setAsOutputPin( GPIO_PORT_P2, GPIO_PIN0 );
	GPIO_setAsPeripheralModuleFunctionInputPin( GPIO_PORT_P2, GPIO_PIN1, GPIO_SECONDARY_MODULE_FUNCTION );
	GPIO_setAsPeripheralModuleFunctionOutputPin( GPIO_PORT_P2, GPIO_PIN0, GPIO_SECONDARY_MODULE_FUNCTION );

	/* Set PJ.4 and PJ.5 for LFXT. */
	GPIO_setAsPeripheralModuleFunctionInputPin(  GPIO_PORT_PJ, GPIO_PIN4 + GPIO_PIN5, GPIO_PRIMARY_MODULE_FUNCTION  );

	// Configure button S1 (P5.6) interrupt and S2 P(5.5)
    GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN6, GPIO_HIGH_TO_LOW_TRANSITION);
    GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN5, GPIO_HIGH_TO_LOW_TRANSITION);
    GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);

	/* Set DCO frequency to 16 MHz. */
	setFrequency(FreqLevel);

	/* Set external clock frequency to 32.768 KHz. */
	CS_setExternalClockSource( 32768, 0 );

	/* Set ACLK = LFXT. */
	CS_initClockSignal( CS_ACLK, CS_LFXTCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Set SMCLK = DCO with frequency divider of 1. */
	CS_initClockSignal( CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Set MCLK = DCO with frequency divider of 1. */
	CS_initClockSignal( CS_MCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Start XT1 with no time out. */
	CS_turnOnLFXT( CS_LFXT_DRIVE_0 );

	/* Disable the GPIO power-on default high-impedance mode. */
	PMM_unlockLPM5();
}
/*-----------------------------------------------------------*/

int _system_pre_init( void )
{
    /* Stop Watchdog timer. */
    WDT_A_hold( __MSP430_BASEADDRESS_WDT_A__ );

    /* Return 1 for segments to be initialised. */
    return 1;
}

/*
 * port 5 interrupt service routine to handle s1 and s2 button press
 *
 */

void __attribute__((__interrupt__(PORT5_VECTOR))) Port_5(void)
{
    GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

    /* Button pushed, do something if you need to */
    uint16_t status_5_5 = GPIO_getInterruptStatus(GPIO_PORT_P5, GPIO_PIN5),
             status_5_6 = GPIO_getInterruptStatus(GPIO_PORT_P5, GPIO_PIN6);
    button_pushed(status_5_5, status_5_6);

    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
}

#ifndef EXTFRAM_USE_DMA
/* The MSP430X port uses this callback function to configure its tick interrupt.
This allows the application to choose the tick interrupt source.
configTICK_VECTOR must also be set in FreeRTOSConfig.h to the correct
interrupt vector for the chosen tick interrupt source.  This implementation of
vApplicationSetupTimerInterrupt() generates the tick from timer A0, so in this
case configTICK_VECTOR is set to TIMER0_A0_VECTOR. */
static void vApplicationSetupTimerInterrupt( void )
{
const unsigned short usACLK_Frequency_Hz = 32768;

    /* Ensure the timer is stopped. */
    TA0CTL = 0;

    /* Run the timer from the ACLK. */
    TA0CTL = TASSEL_1;

    /* Clear everything to start with. */
    TA0CTL |= TACLR;

    /* Set the compare match value according to the tick rate we want. */
    TA0CCR0 = usACLK_Frequency_Hz / configTICK_RATE_HZ;

    /* Enable the interrupts. */
    TA0CCTL0 = CCIE;

    /* Start up clean. */
    TA0CTL |= TACLR;

    /* Up mode. */
    TA0CTL |= MC_1;
}
#endif
/*-----------------------------------------------------------*/

#ifdef __GNUC__
/* GCC hack - when data in .bss and .data is large, initializing them can
 * take a long time and thus the watchdog resets the CPU [1]. This function,
 * which is taken from [3], disables the watchdog before .bss and .data are
 * initialized. How it works is also explained in [3]. MSP430-GCC sources
 * help on understanding this as well [4].
 *
 * As a side note, applications built with msp430-cgt holds WDT during
 * initialization when the linker option "Hold watchdog timer during cinit
 * auto-initialization" is set. Under the hood, `_c_int00_template` calls
 * `_auto_init`, which maps to `__TI_auto_init_nobinit_nopinit_hold_wdt`.
 *
 * References:
 * [1] https://e2e.ti.com/support/microcontrollers/msp-low-power-microcontrollers-group/msp430/f/msp-low-power-microcontroller-forum/948357/msp430-gcc-opensource-watchdog-timeout-during-crt-bss-initialisation
 * [2] https://e2e.ti.com/support/microcontrollers/msp-low-power-microcontrollers-group/msp430/f/msp-low-power-microcontroller-forum/541054/msp430-gcc-startup-source-code
 * [3] https://www.ti.com/lit/ug/slau646f/slau646f.pdf, Sectio 5.3
 * [4] newlib/libgloss/msp430/{crt0.S,memmodel.h}, from msp430-gcc-9.3.1.11-source-full.tar.bz2 available on https://www.ti.com/tool/MSP430-GCC-OPENSOURCE
 * [5] ccs/tools/compiler/ti-cgt-msp430_20.2.5.LTS/lib/src/{boot.c,autoinit.c} under CCS installation
 */
static void __attribute__((naked, used, section(".crt_0042")))
disable_watchdog (void)
{
    WDTCTL = WDTPW | WDTHOLD;
}
#endif
